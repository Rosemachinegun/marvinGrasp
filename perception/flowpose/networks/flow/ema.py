import torch
import logging


class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    Supports periodic updates for training efficiency.
    """

    def __init__(self, parameters, decay, use_num_updates=True, period=1, use_double_precision=True):
        """
        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the result of
                `model.parameters()`.
            decay: The exponential decay (e.g., 0.999).
            use_num_updates: Whether to use number of updates when computing
                averages (adaptive warmup).
            period: Update shadow parameters every N steps (default: 16).
            use_double_precision: Use double precision for numerical stability.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.period = period
        self.step_count = 0
        self.use_double_precision = use_double_precision
        
        # Store shadow parameters (EMA weights)
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        
        # For store/restore functionality
        self.collected_params = []

    def update(self, parameters):
        """
        Update EMA parameters periodically.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object.
        """
        self.step_count += 1
        
        # Only update every `period` steps
        if self.step_count % self.period != 0:
            return
        
        # Compute effective decay compensated for period
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            # Adaptive decay warmup (optional, from ScaleNet)
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        # Compensate for periodic updates
        decay_effective = decay ** self.period
        one_minus_decay = 1.0 - decay_effective
        
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                if self.use_double_precision:
                    # Use double precision to avoid numerical issues
                    delta = param.data.double() - s_param.data.double()
                    s_param_new = s_param.data.double() + one_minus_decay * delta
                    s_param.data.copy_(s_param_new.float())
                else:
                    # Direct update in native precision
                    s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current EMA parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters 
                                 if param.requires_grad]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        """Return state dict for checkpointing."""
        return dict(
            decay=self.decay,
            num_updates=self.num_updates,
            period=self.period,
            step_count=self.step_count,
            shadow_params=self.shadow_params,
            use_double_precision=self.use_double_precision
        )

    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.period = state_dict.get('period', 16)  # Backward compatibility
        self.step_count = state_dict.get('step_count', 0)
        self.shadow_params = state_dict['shadow_params']
        self.use_double_precision = state_dict.get('use_double_precision', True)