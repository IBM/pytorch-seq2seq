import torch

class Optimizer(object):
    """ The Optimizer class encapsulates torch.optim package and provides functionalities
    for learning rate scheduling and gradient norm clipping.

    Args:
        optim_class (torch.optim.Optimizer): optimizer class, e.g. torch.optim.SGD
        max_grad_norm (float, optional): value used for gradient norm clipping,
            set 0 to disable (default 0)
        lr_decay (float, optional): value for learning rate decay:
            lr = lr_decay * lr (default 1)
        decay_after_epoch (float, optional): learning rate starts to decay after the
            specified epoch number, set 0 to disable (default 0)
        **kwargs: arguments for the given optimizer class,
            refer http://pytorch.org/docs/optim.html#algorithms for more information
    """

    _ARG_MAX_GRAD_NORM = 'max_grad_norm'
    _ARG_DECAY_AFTER = "decay_after_epoch"
    _ARG_LR_DECAY = "lr_decay"

    def __init__(self, optim_class, **kwargs):
        self.optim_class = optim_class
        self.optimizer = None
        self.parameters = None

        self.max_grad_norm = self._get_remove(kwargs, Optimizer._ARG_MAX_GRAD_NORM, 0)
        self.lr_decay = self._get_remove(kwargs, Optimizer._ARG_LR_DECAY, 1)
        self.decay_after_epoch = self._get_remove(kwargs, Optimizer._ARG_DECAY_AFTER, 0)
        self.optim_args = kwargs

    def _get_remove(self, args, key, default):
        value = default
        if key in args:
            value = args[key]
            del args[key]
        return value

    def set_parameters(self, parameters):
        """ Set the parameters to optimize.

        Args:
            parameters (iterable): An iterable of torch.nn.Parameter.
        """
        self.parameters = parameters
        self.optimizer = self.optim_class(parameters, **self.optim_args)

    def step(self):
        """ Performs a single optimization step, including gradient norm clipping if necessary. """
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm(self.parameters, self.max_grad_norm)
        self.optimizer.step()

    def update(self, loss, epoch):
        """ Update the learning rate if the conditions are met. Override this method
        to implement your own learning rate schedule.

        Args:
            loss (float): The current loss.  It could be training loss or developing loss
                depending on the caller.  By default the supervised trainer uses developing
                loss.
            epoch (int): The current epoch number.
        """
        after_decay_epoch = self.decay_after_epoch != 0 and epoch >= self.decay_after_epoch
        if after_decay_epoch:
            self.optimizer.param_groups[0]['lr'] *= self.lr_decay

    def load_state_dict(self, optimizer_dict):
        """ Wrapper for loading optimizer state_dict.
            For further reference please refer to http://pytorch.org/docs/master/optim.html#torch.optim.Optimizer.load_state_dict
        """
        self.optimizer.load_state_dict(optimizer_dict)

    def state_dict(self):
        """Wrapper for accessing optimizer state_dict.
            For further reference please refer to http://pytorch.org/docs/master/optim.html#torch.optim.Optimizer.state_dict
        """
        return self.optimizer.state_dict()
