from .base import BasePredictor


def get_predictor(net, brs_mode, device,
                  prob_thresh=0.49,
                  with_flip=False,
                  zoom_in_params=dict(),
                  predictor_params=None,
                  brs_opt_func_params=None,
                  lbfgs_params=None):
    lbfgs_params_ = {
        'm': 20,
        'factr': 0,
        'pgtol': 1e-8,
        'maxfun': 20,
    }

    predictor_params_ = {
        'optimize_after_n_clicks': 1
    }


    zoom_in = None

    if lbfgs_params is not None:
        lbfgs_params_.update(lbfgs_params)
    lbfgs_params_['maxiter'] = 2 * lbfgs_params_['maxfun']

    if brs_opt_func_params is None:
        brs_opt_func_params = dict()

    if predictor_params is not None:
        predictor_params_.update(predictor_params)
    predictor = BasePredictor(net, device, zoom_in=None,  with_flip=with_flip,   **predictor_params_)

    return predictor
