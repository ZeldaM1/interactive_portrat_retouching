import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    # image restoration
    if model == 'dual_auto':  # stage1
        from .DUAL_auto_model import DUAL_auto_model as M
    elif model == 'dual_inter_stage':  # stage2/3
        from .DUAL_inter_stage_model import DUAL_inter_stage_model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
