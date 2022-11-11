Module smoother.models.deconv
=============================
Deconvolute spatial-omics data with spatial loss

Functions
---------

    
`DWLS(backend='pytorch', **kwargs) ‑> smoother.models.deconv.DeconvModel`
:   

    
`LinearRegression(backend='pytorch', **kwargs) ‑> smoother.models.deconv.DeconvModel`
:   

    
`LogNormReg(backend='pytorch', **kwargs) ‑> smoother.models.deconv.DeconvModel`
:   

    
`NNLS(backend='pytorch', **kwargs) ‑> smoother.models.deconv.DeconvModel`
:   

    
`NuSVR(backend='pytorch', **kwargs) ‑> smoother.models.deconv.DeconvModel`
:   

Classes
-------

`DWLSConvex(bias=True, dim_in=1, dim_out=1, nonneg=True, max_weights=8)`
:   Damped weighted least square implemented using cvxpy.
    
    Here the error term for each feature is scaled by its observed value, not the predicted
    value as described in https://www.nature.com/articles/s41467-019-10802-z.
    
    Attributes:
            max_weights (float): The upper limit of the scaling weights for regression loss.

    ### Ancestors (in MRO)

    * smoother.models.deconv.LinearRegressionConvex
    * smoother.models.deconv.DeconvModelConvex
    * smoother.models.deconv.DeconvModel

    ### Methods

    `set_model_loss(self)`
    :   Set DWLS regression loss.

`DWLSTorch(bias=True, dim_in=1, dim_out=1, nonneg=True, max_weights=8)`
:   Damped weighted least square implemented using pytorch.
    
    Here the error term for each feature is scaled by its observed value, not the predicted
    value as described in https://www.nature.com/articles/s41467-019-10802-z.
    
    Attributes:
            max_weights (float): The upper limit of weights.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * smoother.models.deconv.LinearRegressionTorch
    * smoother.models.deconv.DeconvModelTorch
    * torch.nn.modules.module.Module
    * smoother.models.deconv.DeconvModel

    ### Methods

    `get_model_loss_fn(self)`
    :

`DeconvModel()`
:   Interface for convex deconvolution models.

    ### Descendants

    * smoother.models.deconv.DeconvModelConvex
    * smoother.models.deconv.DeconvModelTorch

    ### Methods

    `get_model_loss(self)`
    :   Get regression reconstruction loss.

    `get_props(self)`
    :   Get the predicted celltype abundance from the trained deconvolution model.

    `get_sp_loss(self)`
    :   Get spatial loss (weighted by lambda_spatial_loss).

`DeconvModelConvex(dim_in, dim_out, model_name)`
:   Class for convex deconvolution models implemented using cvxpy.
    
    Attributes:
            dim_in (int): Input dimension, also number of groups.
            dim_out (int): Output dimension, also number of spots.
            model_name (str): Name of the deconvolution model.
            nonneg (bool): Whether to apply nonnegative contraint
                    on spatial variables.
            deconv_configs (dict): Dictionary of deconvolution configurations.
            deconv_time (float): Time spent on deconvolution.

    ### Ancestors (in MRO)

    * smoother.models.deconv.DeconvModel

    ### Descendants

    * smoother.models.deconv.LinearRegressionConvex

    ### Methods

    `deconv(self, x, y, spatial_loss: smoother.losses.SpatialLoss = None, lambda_spatial_loss=0.0, verbose=False, quiet=False, solver=None, **kwargs)`
    :   Solve the regression-based spatial deconvolution problem using cvxpy.
        
        Loss_total = Loss_model + `lambda_spatial_loss` * Loss_spatial
        
        See https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options
        for solver options.
        
        Args:
                x (2D tensor): Bulk feature signiture matrix, num_feature x num_group.
                y (2D tensor): Spatial feature matrix, num_feature x num_spot.
                spatial_loss (SpatialLoss): The spatial smoothing loss.
                        Spatial prior can be one of 'none', 'kl', 'sma', 'sar', 'car', 'icar'
                        - KL : KL divergence of cell type proportion vectors of neiboring spots
                        - SMA : Spatial moving average
                        - SAR : Simultaneous auto-regressive model
                        - CAR : Conditional auto-regressive model
                        - ICAR : Intrinsic conditional auto-regressive model
                        See model description for more details.
                lambda_spatial_loss (float): Specifies the strength of the spatial smoothing.
                        Notice that setting `spatial_loss = None` will be more efficient than
                        setting `lambda_spatial_loss = 0`, but the former won't benefit from
                        warm start (i.e., speedup when only lambda_spatial_loss is changed).
                verbose (bool): If True, print solver output.
                quiet (bool): If True, no output printed, including final message.
                solver (str): The solver to use.
                kwargs: Additional keyword arguments specifying solver specific options.

    `set_params(self, configs) ‑> bool`
    :   Set model parameters. Return True if warm start.

`DeconvModelTorch(dim_in, dim_out, model_name)`
:   Class for deconvolution models implemented using pytorch.
    
    Attributes:
            dim_in (int): Input dimension, also number of groups.
            dim_out (int): Output dimension, also number of spots.
            use_bias (bool): Whether to add bias term in regression.
            model_name (str): Name of the deconvolution model.
            nonneg (bool): Whether to apply nonnegative contraint
                    on spatial variables.
            deconv_configs (dict): Dictionary of deconvolution configurations.
            deconv_time (float): Time spent on deconvolution.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module
    * smoother.models.deconv.DeconvModel

    ### Descendants

    * smoother.models.deconv.LinearRegressionTorch

    ### Methods

    `deconv(self, x, y, spatial_loss: smoother.losses.SpatialLoss = None, lambda_spatial_loss=0.0, top_loss: smoother.losses.TopLoss = None, lambda_top_loss=0.0, lr=0.001, max_epochs=1000, patience=10, tol=1e-05, init_with_lr_sol=True, verbose=True, quiet=False)`
    :   Deconvolute spatial-omics data with spatial loss.
        
        Loss_total = Loss_model + `lambda_spatial_loss` * Loss_spatial +
                `lambda_top_loss` * Loss_topological.
        
        Args:
                x (2D tensor): Bulk feature signiture matrix, num_feature x num_group.
                y (2D tensor): Spatial feature matrix, num_feature x num_spot.
                spatial_loss (SpatialLoss): The spatial smoothing loss.
                        Spatial prior can be one of 'none', 'kl', 'sma', 'sar', 'car', 'icar'
                        - KL : KL divergence of cell type proportion vectors of neiboring spots
                        - SMA : Spatial moving average
                        - SAR : Simultaneous auto-regressive model
                        - CAR : Conditional auto-regressive model
                        - ICAR : Intrinsic conditional auto-regressive model
                        See model description for more details.
                lambda_spatial_loss (float): Specifies the strength of the spatial smoothing.
                top_loss (TopLoss): The topological loss (expected betti number for each group).
                        top_loss.betti_priors = {
                                group_id : {betti_k : expected number of barcodes (prior)}, ...
                        }
                lambda_top_loss (float): Specifies the strength of the topological constraints.
                lr (float): Learning rate.
                max_epochs (int): Maximum number of training epochs. If -1, iterate until
                        convergence (`d_loss` < 1e-5).
                patient (int): Number of epochs to wait for the loss to decrease before stopping.
                tol (float): Tolerance of loss convergence.
                init_with_lr_sol (bool): Whether to initialize regression weights with the OLE solution.
                verbose (bool): If True, print out loss while training.
                quiet (bool): If True, no output printed.

    `final_sanity_check(self)`
    :   Make sure constraints are followed after deconvolution.

    `get_top_loss(self)`
    :   Get toplogical loss (weighted by lambda_top_loss).

    `init_with_lr_sol(self, x, y)`
    :   Initialize model with linear regression solution.

    `set_params(self, dim_in, dim_out, bias)`
    :

`LinearRegressionConvex(bias=True, dim_in=1, dim_out=1)`
:   Linear regression implemented using cvxpy.

    ### Ancestors (in MRO)

    * smoother.models.deconv.DeconvModelConvex
    * smoother.models.deconv.DeconvModel

    ### Descendants

    * smoother.models.deconv.DWLSConvex
    * smoother.models.deconv.NNLSConvex
    * smoother.models.deconv.NuSVRConvex

    ### Methods

    `get_model_loss(self)`
    :   Get model regression reconstruction loss.

    `set_lambda_only(self, configs) ‑> bool`
    :   Check if the same configuration is used except lambda_spatial_loss.
        
        If True, will only update lambda_spatial_loss and enable warm start.

    `set_model_loss(self)`
    :   Set model reconstruction loss.

    `set_sp_loss(self)`
    :   Set spatial loss.

`LinearRegressionTorch(bias=True, dim_in=1, dim_out=1)`
:   Linear regression implemented using pytorch.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * smoother.models.deconv.DeconvModelTorch
    * torch.nn.modules.module.Module
    * smoother.models.deconv.DeconvModel

    ### Descendants

    * smoother.models.deconv.DWLSTorch
    * smoother.models.deconv.LogNormRegTorch
    * smoother.models.deconv.NNLSTorch
    * smoother.models.deconv.NuSVRTorch

    ### Methods

    `final_sanity_check(self)`
    :   Make sure constraints are satisfied.

    `forward(self, x) ‑> Callable[..., Any]`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `get_model_loss(self)`
    :   Calculate regression loss.

    `get_model_loss_fn(self)`
    :

    `get_sp_loss(self)`
    :   Calculate spatial loss (weighted by lambda_spatial_loss).

    `get_top_loss(self)`
    :   Calculate topological loss (weighted by lambda_top_loss).

    `set_params(self, dim_in, dim_out, bias=True) ‑> None`
    :

`LogNormRegTorch(bias=True, epsilon=1, dim_in=1, dim_out=1)`
:   Log-normal deconvolution implemented using pytorch.
    
    Minimize MSE(log(Y), log(X @ W)). Here Y and X are all in the raw count space.
    
    Instead of minimizing the least square loss, this model will minimize MSE after
    log-transforming both observation and prediction, as described in
    in https://www.nature.com/articles/s41467-022-28020-5 as Algorithm 1.
    
    Attributes:
            epsilon (float): pseudo count added before log transformation.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * smoother.models.deconv.LinearRegressionTorch
    * smoother.models.deconv.DeconvModelTorch
    * torch.nn.modules.module.Module
    * smoother.models.deconv.DeconvModel

    ### Methods

    `get_model_loss_fn(self)`
    :

`NNLSConvex(bias=True, dim_in=1, dim_out=1)`
:   Non-negative least square implemented using cvxpy.

    ### Ancestors (in MRO)

    * smoother.models.deconv.LinearRegressionConvex
    * smoother.models.deconv.DeconvModelConvex
    * smoother.models.deconv.DeconvModel

`NNLSTorch(bias=True, dim_in=1, dim_out=1)`
:   Non-negative least square implemented using pytorch.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * smoother.models.deconv.LinearRegressionTorch
    * smoother.models.deconv.DeconvModelTorch
    * torch.nn.modules.module.Module
    * smoother.models.deconv.DeconvModel

`NuSVRConvex(bias=True, dim_in=1, dim_out=1, C=1.0, nu=0.1, nonneg=False, loss_mode='l2')`
:   Nu-Support vector regression implemented using cvxpy.
    
    Attributes:
            C (float): SVR regularization parameter.
            nu (float): Nu-SVR parameter.
            epsilon (float): Epsilon in the epsilon-SVR model, specifying the the epsilon-tube within which
                    no penalty is associated in the training loss function with points predicted within a distance
                    epsilon from the actual value.
            loss_mode (str): Specifies the regression loss function to use. Either 'l2' or 'l1'.
    
    Initialize a Nu-SVR deconvolution model.
    
    Args:
            C (float): SVR regularization parameter.
            nu (float): Nu-SVR parameter.
            nonneg (bool): Whether to require nonnegative regression coefficients.
            loss_mode (str): Specifies the regression loss function to use. Either 'l2' or 'l1'.

    ### Ancestors (in MRO)

    * smoother.models.deconv.LinearRegressionConvex
    * smoother.models.deconv.DeconvModelConvex
    * smoother.models.deconv.DeconvModel

    ### Methods

    `set_model_loss(self)`
    :   Set Nu-SVR regression loss.

`NuSVRTorch(bias=True, dim_in=1, dim_out=1, C=1.0, nu=0.1, nonneg=False, loss_mode='l2')`
:   Nu-Support vector regression implemented using pytorch.
    
    Attributes:
            C (float): SVR regularization parameter.
            nu (float): Nu-SVR parameter.
            epsilon (float): Epsilon in the epsilon-SVR model, specifying the the epsilon-tube within which
                    no penalty is associated in the training loss function with points predicted within a distance
                    epsilon from the actual value.
            loss_mode (str): Specifies the regression loss function to use. Either 'l2' or 'l1'.
    
    Initialize a Nu-SVR deconvolution model.
    
    Args:
            C (float): SVR regularization parameter.
            nu (float): Nu-SVR parameter.
            nonneg (bool): Whether to require nonnegative regression coefficients.
            loss_mode (str): Specifies the regression loss function to use. Either 'l2' or 'l1'.

    ### Ancestors (in MRO)

    * smoother.models.deconv.LinearRegressionTorch
    * smoother.models.deconv.DeconvModelTorch
    * torch.nn.modules.module.Module
    * smoother.models.deconv.DeconvModel

    ### Methods

    `get_model_loss_fn(self)`
    :