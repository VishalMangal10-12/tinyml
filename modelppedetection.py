'''
  PPE Model Implementation
'''
import cv2
import numpy as np
import torch
import json
import torch.nn.functional as F
from math import sqrt
from utils_new import *
import time

from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc

from modelbase import ModelBase

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ppe detection model
class PPEDetection(ModelBase):
    def __init__(self, modelname='qualityinspection', threshold=0.5):
        super(PPEDetection, self).__init__(modelname, threshold)
        self.format = mc.ModelInput.FORMAT_NONE
        self.request = service_pb2.ModelInferRequest()
        self.request.model_name = self.model_name
        self.request.model_version = '1'

        output0 = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
        output0.name = 'output'
        output1 = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
        output1.name = 'output_1'
        self.request.outputs.extend([output0, output1])

        input = service_pb2.ModelInferRequest().InferInputTensor()
        input.name = 'input'
        input.datatype = 'FP32'
        input.shape.extend([1, 3, 300, 300])

        self.request.ClearField("inputs")
        self.request.inputs.extend([input])

        # Prior boxes
        self.priors_cxcy = self.__create_prior_boxes()
        self.voc_labels = ('pcb', 'switchmiss', 'componentmiss')
        self.label_map = {k: v + 1 for v, k in enumerate(self.voc_labels)}
        self.label_map['background'] = 0
        self.rev_label_map = {v: k for k, v in self.label_map.items()}  # Inverse mapping
        self.n_classes = 4

    def __preprocess(self):
        logger.info("Preprocessing...")
        # Transform
        image = cv2.resize(self.image,(300,300))
        # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
        image = (image - image.mean(axis=(0, 1, 2), keepdims=True)) / image.std(axis=(0, 1, 2),
                                                                                                keepdims=True)
        image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
        image = image.transpose((2, 0, 1))
        # Convert PIL image to Torch tensor
        image = torch.Tensor(image)
        # Move to default device
        image = image.to("cpu")
        image = image.unsqueeze(0)
        data = json.dumps({'data': image.tolist()})
        data = np.array(json.loads(data)['data']).astype('float32')
        self.transimage = data
        #print('PP_END')

    def __create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        return prior_boxes

    def __detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        #print('D1')
        batch_size = predicted_locs.size(0)
        #print('D11')
        n_priors = self.priors_cxcy.size(0)
        #print('D12')
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)
        #print('D2')

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
        #print('D3')

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates
            #print('D4')

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            #max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.n_classes):
                #print('D5')

                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                #print('D6')
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)
                #print('D7')

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    condition = overlap[box] > max_overlap
                    condition = torch.tensor(condition, dtype=torch.uint8).to(device)
                    suppress = torch.max(suppress, condition)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size

    def __postprocess(self):
        logger.info("Postprocessing...")
        output_results = []
        index = 0
        for output in self.response.outputs:
            shape = []
            for value in output.shape:
                shape.append(value)
            output_results.append(
                np.frombuffer(self.response.raw_output_contents[index], dtype=np.float32))
            output_results[-1] = np.resize(output_results[-1], shape)
            index += 1

        predicted_scores = output_results[0].astype('float32')
        predicted_locs = output_results[1].astype('float32')

        predicted_locs = torch.Tensor(predicted_locs)
        predicted_scores = torch.Tensor(predicted_scores)

        #print(predicted_locs)
        #print(predicted_scores)

        #print('PP_1')
        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = self.__detect_objects(predicted_locs, predicted_scores, min_score=0.3,
                                                                 max_overlap=0.5, top_k=200)
        #print('PP_2')
        # Move detections to the CPU
        det_boxes = det_boxes[0].to('cpu')
        height,width,channels = self.image.shape
        #print('PP_3')
        # Transform to original image dimensions
        original_dims = torch.FloatTensor(
            [width, height, width, height]).unsqueeze(0)
        #print('PP_4')
        det_boxes = det_boxes * original_dims
        # Decode class integer labels
        det_labels = [self.rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
        # import pdb;pdb.set_trace()
        # for i in range(len(det_boxes)):
        #     x1,y1 = det_boxes[i][0],det_boxes[i][1]
        #     x2, y2 = det_boxes[i][2], det_boxes[i][3]
        #     draw_img = cv2.rectangle(original_image.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)
        # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
        print(self.rev_label_map)
        print(det_labels)
        print(det_scores)
        if det_labels == ['background']:
            # Just return original image
            print("No PCB Detected!!!!")
            cv2.putText(self.image, "NO PCB", (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 4)
            return 0
        else:
            if "pcb" in det_labels:
                idx = det_labels.index("pcb")
                if not float(det_scores[0][idx])>0.5:
                    print("No PCB Detected!!!!")
                    cv2.putText(self.image, "NO PCB", (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 4)
                    return 0
                else:
                    print("PCB Detected !!!")
                    cv2.putText(self.image, "PCB", (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 4)
                    if "switchmiss" in det_labels or "componentmiss" in det_labels:
                        if "switchmiss" in det_labels:
                            print("Defect Identified with Switches")
                            cv2.putText(self.image, "PCB - Switch Missed", (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 4)
                        if "componentmiss" in det_labels:
                            print("Defect Identified with Components")
                            cv2.putText(self.image, "PCB - Component Missed", (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 4)
                    else:
                        print("No Defects Found!!")
                        cv2.putText(self.image, "PCB - Good", (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 4)
            else:
                print("No PCB Identified!!!")
                cv2.putText(self.image, "NO PCB", (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 4)
                return 0

    def predict(self, image):      
        self.image = image
        self.__preprocess()
        start_time = time.time()
        self.run_infer()
        logger.debug("Model {} Inference time {:.2f} msecs".format(self.model_name, (time.time()-start_time)*1000))
        self.__postprocess()
        image = self.image
        return True

    def getlastevent(self):
        return True

