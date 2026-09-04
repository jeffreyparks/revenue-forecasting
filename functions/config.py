# Configuration for the forecasting pipeline
#
# Change these values to adjust model behavior.
# The RESPONSE_VAR setting controls which target variable is used for training
# and predictions. Historical options include 'nd' (net demand) and 'visits'.

# Response variable for modeling
# Options: 'nd' (net demand), 'visits' (site visits), or other future targets
RESPONSE_VAR = 'nd'

# All known response variables — excluded from features regardless of which is the active target
RESPONSE_VARS = ['nd', 'visits']
