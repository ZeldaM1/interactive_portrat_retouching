import tkinter as tk
from tkinter import messagebox, filedialog, ttk

import cv2
import numpy as np
from PIL import Image

from interactive_demo.canvas import CanvasImage
from interactive_demo.controller import InteractiveController
from interactive_demo.wrappers import BoundedNumericalEntry, FocusHorizontalScale, FocusCheckButton, \
    FocusButton, FocusLabelFrame
import data.util as util

class InteractiveDemoApp(ttk.Frame):
    def __init__(self, master, args, model):
        super().__init__(master)
        self.master = master
        master.title("Region-Aware Portrait Retouching")
        master.withdraw()
        master.update_idletasks()
        x = (master.winfo_screenwidth() - master.winfo_reqwidth()) / 2
        y = (master.winfo_screenheight() - master.winfo_reqheight()) / 2
        master.geometry("+%d+%d" % (x, y))
        self.pack(fill="both", expand=True)

        self.brs_modes = ['NoBRS']
        self.limit_longest_size = args.limit_longest_size

        self.controller = InteractiveController(model, args.device,
                                                predictor_params={'brs_mode': 'NoBRS'},
                                                update_image_callback=self._update_image)

        self._init_state()
        self._add_canvas()
        self._add_buttons()

        master.bind('<space>', lambda event: self.controller.finish_object())
        master.bind('a', lambda event: self.controller.partially_finish_object())

        self.state['zoomin_params']['skip_clicks'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['target_size'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['expansion_ratio'].trace(mode='w', callback=self._reset_predictor)
        self.state['predictor_params']['net_clicks_limit'].trace(mode='w', callback=self._change_brs_mode)
        self.state['lbfgs_max_iters'].trace(mode='w', callback=self._change_brs_mode)
        self._change_brs_mode()

    def _init_state(self):
        self.state = {
            'zoomin_params': {
                'use_zoom_in': tk.BooleanVar(value=True),
                'fixed_crop': tk.BooleanVar(value=True),
                'skip_clicks': tk.IntVar(value=-1),
                'target_size': tk.IntVar(value=self.limit_longest_size),
                'expansion_ratio': tk.DoubleVar(value=1.4)
            },

            'predictor_params': {
                'net_clicks_limit': tk.IntVar(value=8)
            },
            'brs_mode': tk.StringVar(value='NoBRS'),
            'prob_thresh': tk.DoubleVar(value=0.5),
            'lbfgs_max_iters': tk.IntVar(value=20),

            'alpha_blend': tk.DoubleVar(value=0.5),
            'click_radius': tk.IntVar(value=3),
        }


    def _add_canvas(self):

        self.canvas_frame = FocusLabelFrame(self, text="Image")
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0, cursor="hand1", width=400, height=400)
        self.canvas.grid(row=0, column=0, sticky='nswe', padx=5, pady=5)

        self.image_on_canvas = None
        self.canvas_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)

    def _add_buttons(self):

        self.menubar = FocusLabelFrame(self)  # , bd=1)
        self.menubar.pack(side=tk.TOP,fill='x', padx=5, pady=10)

        button = FocusButton(self.menubar, text='Load image', command=self._load_image_callback)
        button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.save_image_btn = FocusButton(self.menubar, text='Save image', command=self._save_image_callback)
        self.save_image_btn.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.save_image_btn.configure(state=tk.DISABLED)

        button = FocusButton(self.menubar, text='About', command=self._about_callback)
        button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        button = FocusButton(self.menubar, text='Exit', command=self.master.quit)
        button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        #---------------------------------------------------------
        self.control_frame = FocusLabelFrame(self, text="Controls")
        self.control_frame.pack(side=tk.TOP, fill='x', padx=5, pady=3)
        master = self.control_frame

        self.clicks_options_frame = FocusLabelFrame(master, text="Clicks management")
        self.clicks_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)

        self.run_click_button = FocusButton(self.clicks_options_frame, text='Run', bg='#b6d7a8', fg='black', width=10, height=2,
                                             state=tk.DISABLED, command=self.controller.run_click)
        self.run_click_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)



        self.undo_click_button =  FocusButton(self.clicks_options_frame, text='Undo click', bg='#ffe599', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self.controller.undo_click)
        self.undo_click_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.reset_clicks_button = \
            FocusButton(self.clicks_options_frame, text='Reset clicks', bg='#ea9999', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self._reset_last_object)
        self.reset_clicks_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        self.target_size_frame = FocusLabelFrame(master, text="inference size")
        self.target_size_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.target_size_frame, from_=400, to=1200, resolution=1, command=self._reset_predictor,
                             variable=self.state['zoomin_params']['target_size']).pack(padx=10, anchor=tk.CENTER)


        self.click_radius_frame = FocusLabelFrame(master, text="Visualisation click radius")
        self.click_radius_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.click_radius_frame, from_=0, to=7, resolution=1, command=self._update_click_radius,
                             variable=self.state['click_radius']).pack(padx=10, anchor=tk.CENTER)






    def _load_image_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*"),
            ], title="Chose an image")

            if len(filename) > 0:
                # image = util.read_img(None, filename)
                image = cv2.imread(filename)
                image = image[:, :, [2, 1, 0]]
                self.controller.set_image(image)
                self.save_image_btn.configure(state=tk.NORMAL)






    def _save_image_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):

            output = self.controller.probs_history[-1]
            if output is None:
                return

            output_click = self.controller.get_visualization(output, alpha_blend=self.state['alpha_blend'].get(),
                                                      click_radius=self.state['click_radius'].get())
            # output_click = cv2.cvtColor(output_click, cv2.COLOR_BGR2RGB)

            filename = filedialog.asksaveasfilename(parent=self.master, initialfile='output.png', filetypes=[
                ("PNG image", "*.png"),
                ("BMP image", "*.bmp"),
                ("All files", "*.*"),
            ], title="Save the current mask as...")

            if len(filename) > 0:

                output_all=np.hstack([self.controller.probs_history[0], output_click])
                output_all = cv2.cvtColor(output_all,cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, output_all)



    def _about_callback(self):
        self.menubar.focus_set()

        text = [
            "Developed by:",
            "K.Sofiiuk and I. Petrov",
            "The MIT License, 2021"
        ]

        messagebox.showinfo("About Demo", '\n'.join(text))

    def _reset_last_object(self):
        self.state['alpha_blend'].set(0.5)
        self.state['prob_thresh'].set(0.5)
        self.controller.reset_last_object()

    def _update_click_radius(self, *args):
        if self.image_on_canvas is None:
            return
        if len(self.controller.probs_history)>1:
            self._update_image(self.controller.probs_history[-1])
        else:
            self._update_image(self.controller.image)

    def _change_brs_mode(self, *args):

        self._reset_predictor()

    def _reset_predictor(self, *args, **kwargs):
        brs_mode = self.state['brs_mode'].get()
        prob_thresh = self.state['prob_thresh'].get()
        net_clicks_limit = None if brs_mode == 'NoBRS' else self.state['predictor_params']['net_clicks_limit'].get()

        if self.state['zoomin_params']['use_zoom_in'].get():
            zoomin_params = {
                'skip_clicks': self.state['zoomin_params']['skip_clicks'].get(),
                'target_size': self.state['zoomin_params']['target_size'].get(),
                'expansion_ratio': self.state['zoomin_params']['expansion_ratio'].get()
            }
            if self.state['zoomin_params']['fixed_crop'].get():
                zoomin_params['target_size'] = (zoomin_params['target_size'], zoomin_params['target_size'])
        else:
            zoomin_params = None
        predictor_params = {
            'brs_mode': brs_mode,
            'prob_thresh': prob_thresh,
            'zoom_in_params': zoomin_params,
            'predictor_params': {
                'net_clicks_limit': net_clicks_limit,
                'max_size': self.state['zoomin_params']['target_size'].get()
            },
            'brs_opt_func_params': {'min_iou_diff': 1e-3},
            'lbfgs_params': {'maxfun': self.state['lbfgs_max_iters'].get()}
        }
        self.controller.reset_predictor(predictor_params)

    def _click_callback(self, is_positive, x, y):
        self.canvas.focus_set()

        if self.image_on_canvas is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        if self._check_entry(self):
            self.controller.add_click(x, y, is_positive)

    def _update_image(self, pred, reset_canvas=False):
        image = self.controller.get_visualization(pred,alpha_blend=self.state['alpha_blend'].get(),
                                                  click_radius=self.state['click_radius'].get())
        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas)
            self.image_on_canvas.register_click_callback(self._click_callback)

        self._set_click_dependent_widgets_state()
        if image is not None:
            self.image_on_canvas.reload_image(Image.fromarray(image), reset_canvas)


    def _set_click_dependent_widgets_state(self):
        after_1st_click_state = tk.NORMAL if self.controller.is_incomplete_mask else tk.DISABLED
        before_1st_click_state = tk.DISABLED if self.controller.is_incomplete_mask else tk.NORMAL
        self.run_click_button.configure(state=after_1st_click_state)
        self.undo_click_button.configure(state=after_1st_click_state)
        self.reset_clicks_button.configure(state=after_1st_click_state)

    def _check_entry(self, widget):
        all_checked = True
        if widget.winfo_children is not None:
            for w in widget.winfo_children():
                all_checked = all_checked and self._check_entry(w)

        if getattr(widget, "_check_bounds", None) is not None:
            all_checked = all_checked and widget._check_bounds(widget.get(), '-1')

        return all_checked
