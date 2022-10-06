Module smoother.models.impute
=============================
Impute spatial feature of interest with spatial loss

Classes
-------

`ImputeConvex(y_obs, spatial_loss_all: smoother.models.losses.SpatialLoss, fixed_obs=True, nonneg=False, **kwargs)`
:   Imputation solver implemented using cvxpy.
    
    Attributes:
            var_obs : Observed spatial feature matrix, n_obs x n_feature.
            var_missing : Missing spatial feature matrix to impute, n_missing x n_feature.
            var_all : Concatenated observed and missing spatial feature matrix, n_all x n_feature.
            recon_loss_exp (cp.Expression): Reconstruction loss expression.
            sp_loss_exp (cp.Expression): Spatial loss expression.
            See super class attributes for additional arguments.
    
    Initialize and run the imputation.
    
    Args:
            y_obs (2D tensor): Observed spatial feature matrix, n_obs x n_feature.
            spatial_loss_all (SpatialLoss): Spatial loss object build from the combined
                                    spatial coordinates containing both observed (the first n_obs rows)
                                    and missing spots (the rest rows).
            See class attributes for additional arguments.

    ### Ancestors (in MRO)

    * smoother.models.impute.ImputeModel

    ### Methods

    `get_recon_loss(self)`
    :   Calculate loss of deviation from observation.
        
        If the observed data is not fixed, then the loss is used to make sure data does not deviate
        too much from the observed value after imputation.

    `get_results(self)`
    :   Get spatial features of all spots after imputation.
        
        The first n_obs spots are the observed spots and the rest are the missing ones.
        If fixed_obs is True, then the returned observed data is not updated.

    `get_sp_loss(self)`
    :   Calculate loss of spatial smoothness (weighted by lambda_spatial_loss).

    `impute(self, lambda_spatial_loss=1.0, verbose=False, quiet=False, solver=None, **kwargs) ‑> bool`
    :   Run imputation algorithm.
        
        Args:
                lambda_spatial_loss (float): Specifies the strength of the spatial smoothing.
                verbose (bool): If True, print out loss while training.
                quiet (bool): If True, no output printed.
        
        Returns:
                bool: True if imputation is successful.

    `set_recon_loss(self)`
    :   Set loss expression of deviation from observation.
        
        If the observed data is not fixed, then the loss is used to make sure data does not deviate
        too much from the observed value after imputation.

    `set_sp_loss(self)`
    :   Set spatial loss expression (before weighted by lambda_spatial_loss).

    `set_variables(self)`
    :   Initialize parameters to estimate.

`ImputeModel(y_obs, spatial_loss_all: smoother.models.losses.SpatialLoss, fixed_obs=True, nonneg=False, **kwargs)`
:   Class for different implementations of spatial imputation.
    
    This class is designed for coorperative inheritance (mixin).
    
    Attributes:
            y_obs (2D tensor): Observed spatial feature matrix, n_obs x n_feature.
            spatial_loss_all (SpatialLoss): Spatial loss object build from the combined
                    spatial coordinates containing both observed (the first n_obs rows)
                    and missing spots (the rest rows).
            fixed_obs (bool): Whether to allow updates in the observed data.
            nonneg (bool): Whether to enforce nonnegativity on spatial features.
            n_feature (int): Number of spatial features to process.
            n_all (int): Number of total spots. n_all = n_obs + n_missing.
            n_obs (int): Number of observed spots.
            n_missing (int): Number of missing spots.
            var_obs : Observed spatial feature matrix, n_obs x n_feature.
            var_missing : Missing spatial feature matrix to impute, n_missing x n_feature.
            var_all : Concatenated observed and missing spatial feature matrix, n_all x n_feature.
            impute_configs (dict): Dictionary of imputation configurations.
            impute_time (float): Time spent on deconvolution.
            impute_flag (bool): Whether the imputation was successful.
    
    Initialize ImputeModel object.

    ### Descendants

    * smoother.models.impute.ImputeConvex
    * smoother.models.impute.ImputeTorch

    ### Methods

    `get_recon_loss(self)`
    :   Get reconstruction loss.

    `get_results(self)`
    :   Get spatial features after imputation.

    `get_sp_loss(self)`
    :   Get spatial loss (weighted by lambda_spatial_loss).

    `set_params(self, y_obs, spatial_loss_all, fixed_obs=True, nonneg=False)`
    :   Set configuration parameters and initialize variables.

    `set_variables(self)`
    :   Set variables to impute.

`ImputeTorch(y_obs, spatial_loss_all: smoother.models.losses.SpatialLoss, fixed_obs=True, nonneg=False, **kwargs)`
:   Imputation solver implemented using pytorch.
    
    Attributes:
            var_obs (2D tensor): Observed spatial feature matrix, n_obs x n_feature.
            var_missing (2D tensor): Missing spatial feature matrix to impute, n_missing x n_feature.
            var_all (2D tensor): Concatenated observed and missing spatial feature matrix, n_all x n_feature.
            See super class attributes for additional arguments.
    
    Initialize and run the imputation.
    
    Args:
            y_obs (2D tensor): Observed spatial feature matrix, n_obs x n_feature.
            spatial_loss_all (SpatialLoss): Spatial loss object build from the combined
                                    spatial coordinates containing both observed (the first n_obs rows)
                                    and missing spots (the rest rows).
            See class attributes for additional arguments.

    ### Ancestors (in MRO)

    * smoother.models.impute.ImputeModel
    * torch.nn.modules.module.Module

    ### Methods

    `final_sanity_check(self)`
    :

    `forward(self, y_obs, **kwargs) ‑> Callable[..., Any]`
    :   Run imputation for a new set of spatial features.

    `get_recon_loss(self)`
    :   Calculate loss of deviation from observation.
        
        If the observed data is not fixed, then the loss is used to make sure data does not deviate
        too much from the observed value after imputation.

    `get_results(self)`
    :   Get spatial features of all spots after imputation.
        
        The first n_obs spots are the observed spots and the rest are the missing ones.
        If fixed_obs is True, then the returned observed data is not updated.

    `get_sp_loss(self)`
    :   Calculate loss of spatial smoothness.

    `impute(self, lambda_spatial_loss=1.0, lr=0.001, max_epochs=1000, patience=10, tol=1e-05, verbose=True, quiet=False) ‑> bool`
    :   Run imputation algorithm.
        
        Args:
                lambda_spatial_loss (float): Specifies the strength of the spatial smoothing.
                lr (float): Learning rate.
                max_epochs (int): Maximum number of training epochs. If -1, iterate until
                        convergence (`d_loss` < tol).
                patient (int): Number of epochs to wait for the loss to decrease before stopping.
                tol (float): Tolerance of loss convergence.
                verbose (bool): If True, print out loss while training.
                quiet (bool): If True, no output printed.
        
        Returns:
                bool: True if imputation is successful.

    `set_variables(self)`
    :   Initialize torch parameters to estimate.