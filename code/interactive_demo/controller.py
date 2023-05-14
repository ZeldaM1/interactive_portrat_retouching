import torch
import numpy as np
from tkinter import messagebox

from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks


class InteractiveController:
    def __init__(self, net, device, predictor_params, update_image_callback, prob_thresh=0.5):
        self.net = net
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None


        self.image = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.reset_predictor()

    def set_image(self, image):
        self.image = image
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        self.probs_history.append(image)
        self.update_image_callback(image,reset_canvas=True)





    def add_click(self, x, y, is_positive):
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
        })

        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)
        self.update_image_callback(self.probs_history[-1])

    def run_click(self):
        pred = self.predictor.get_prediction(self.clicker)

        torch.cuda.empty_cache()

        self.probs_history.append(pred)
        self.update_image_callback(pred)

    def undo_click(self):
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
        if not self.probs_history:
            self.reset_init_image()
        self.probs_history.pop()
        self.update_image_callback(self.probs_history[-1])




    def partially_finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.probs_history.append(pred)
        self.states.append(self.states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_image()
        self.update_image_callback()

        self.object_count += 1
        self.reset_last_object()

    def reset_last_object(self, update_image=True):
        self.states = []

        self.probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_image()
        if update_image:
            self.update_image_callback(self.image)


    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    def reset_init_image(self):
        self.clicker.click_indx_offset = 0

    @property
    def current_object_prob(self):
        if self.probs_history:
            return self.probs_history[-1]
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0


    def get_visualization(self, pred=None, alpha_blend=None, click_radius=None):
        vis = draw_with_blend_and_clicks(pred, alpha=alpha_blend, clicks_list=self.clicker.clicks_list, radius=click_radius)

        return vis



