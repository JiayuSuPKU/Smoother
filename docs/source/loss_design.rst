General spatial loss design
============================

Remember that the spatial loss represents our prior belief on the spatial dependency of the data. 
For different types of data, we may have different beliefs especially on the dependency strength. 
For example, solid tumor tissues tend to be more spatially heterogeneous, therefore we may a weaker spatial regularization.
In this tutorial, we will show how to check the patterns in the data and design a spatial loss tailored to your need.

.. nbgallery::
  tutorials/tutorial_plot_dependency
  tutorials/tutorial_sp_loss_design