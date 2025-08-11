# Packages and VE ---------------------------------------------------------

# Load some useful functions and packages
source("preamble.R")

#Instigate packages required for deep learning + virtual environment
packages = c("keras", "tensorflow","reticulate")
package.check <- lapply(packages, FUN = function(x) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x, dependencies = TRUE)
    library(x, character.only = TRUE)
  }
})

# Load virtual environment created for project
reticulate::use_virtualenv("DeepSPAR_env", required = T)

#Call tensorflow session 
sess = tf$compat$v1$keras$backend$get_session()
sess$list_devices()

# Set tensorflow seed for reproducibility - https://keras3.posit.co/reference/set_random_seed.html
set_random_seed(1)

# Specifying tuning parameters --------------------------------------------

# Specify the number of units in the hidden layers of the quantile regression neural network
quant.nunits = c(16,16,16)

# Specify the quantile level 
quant.level = 0.9

# Specify the number of units in the hidden layers of the GPD neural network
gpd.nunits = c(16,16,16)

# Specify a vector of decreasing learning rates. The first rate will be used for the main training procedure, then subsequentially decreased when near the minimum
l.r.vec = c(0.001, 5e-04,  1e-04)

# Flag to indicate whether to fit the quantile regression model. If you have already fitted it to data, set this to F
run_QR = T

# Flag to indicate whether to fit the GPD model. If you have already fitted it to data, set this to F
run_GPD = T

# Load in and normalise data --------------------------------------------------------------------

df <- read.csv("Data/wind_wave_data.csv")

# Select columns of metocean variables 
X <- as.matrix(df[, c(5:6, 7:8, 9)])  

# Extract variable names 
wave.names = colnames(X)

# Move wave period to log scale
X[, 5] <- log(X[, 5])

# Calculate means for each variable
means = apply(X, 2, mean)

# For the first 4 variables, we centre around 0 (not the mean)
means[1:4]=0

# Calculate sd's for each variable
sds = apply(X, 2, sd)

# Move data to a standardised scale with a star-shaped centre at (0,0,0,0,0)
X.norm = X
for (i in 1:ncol(X)) X.norm[, i] = (X[, i] - means[i])/sds[i]

# Fit the SPAR model using neural networks ---------------------------------------------------------------

# Compute polar coordinates - see https://en.wikipedia.org/wiki/N-sphere
polar = rect2polar(t(X.norm)) 

# Compute sample size 
n = dim(X.norm)[1]

# Compute dimension
d = dim(X.norm)[2]

# Create angular-radial decomp 
R = polar$r
W = X.norm/R
W = as.matrix(W)

# Make 10% validation set and 10% test set 
set.seed(1)
valid.inds = sample(1:n, round(n/10))
test.inds = sample((1:n)[-valid.inds], round(n/10))

# Specify activation function for neural networks
activation.func = "relu"

# Compute training, validation and test sets of radial and angular variables 
R.train <- R[-valid.inds]
W.train <- W[-valid.inds,]
R.valid <- R[valid.inds]
W.valid <- W[valid.inds,]
R.test <- R[test.inds]
W.test <- W[test.inds,]

# Only fit model if flag is set to T
if(run_QR){
  
  # Specify input layer
  # This is so that Keras knows what data shape/dimensions to expect.
  input_pseudo_angles <- layer_input(shape = d, name = "input_pseudo_angles")
  
  # Specify a densely-connected MLP
  # We define a ReLU neural network with exponential activation in the final layer (to ensure that the radial quantile is strictly positive)
  qBranch <- input_pseudo_angles %>%
    layer_dense(units = quant.nunits[1], activation = activation.func, name = "q_dense1", kernel_regularizer = regularizer_l1_l2(l1 = 1e-04,
                                                                                                                                 l2 = 1e-04))
  for (i in 2:length(quant.nunits)) {
    qBranch <- qBranch %>%
      layer_dense(units = quant.nunits[i], activation = activation.func, name = paste0("q_dense", i), kernel_regularizer = regularizer_l1_l2(l1 = 1e-04,
                                                                                                                                             l2 = 1e-04))
  }
  
  qBranch <- qBranch %>%
    layer_dense(units = 1, activation = "exponential", name = "q_final", kernel_regularizer = regularizer_l1_l2(l1 = 1e-04,
                                                                                                                l2 = 1e-04))  
  # Construct Keras model
  quant.model <- keras_model(inputs = c(input_pseudo_angles), outputs = c(qBranch))
  summary(quant.model)
  
  # Compile the model with the tilted/pinball loss function and the adam optimiser
  quant.model %>%
    compile(optimizer = "adam", loss = tilted_loss, run_eagerly = T)
  
  # After every epoch, we use a checkpoint to save the weights. 
  # Only the current best version of the model is saved, i.e., the one that minimises the loss evaluated on the validation data
  checkpoint <- callback_model_checkpoint(filepath = paste0("runs/QR_est/qr_fit"), monitor = "val_loss", verbose = 0,
                                          save_best_only = TRUE, save_weights_only = TRUE, mode = "min", save_freq = "epoch")
  
  # Set number of epochs for training
  n.epochs <- 100  
  
  # Set batch size for training
  batch.size <- 512
  
  # Convert vectors to matrices. Keras wants every object to be in an array/matrix format
  dim(R)=c(length(R),1); dim(R.train)=c(length(R.train),1); dim(R.valid)=c(length(R.valid),1);dim(R.test)=c(length(R.test),1)
  
  # Train the Keras model. Loss values will be stored in history object.
  history <- quant.model %>%
    fit(list(W.train), R.train, epochs = n.epochs, batch_size = batch.size, 
        callback = list(checkpoint,callback_early_stopping(monitor = "val_loss", min_delta = 0, patience = 5)), 
        validation_data = list(list(input_pseudo_angles = W.valid),R.valid))
  
  # Load the best fitting model from the checkpoint save
  quant.model <- load_model_weights_tf(quant.model, filepath = paste0("runs/QR_est/qr_fit"))
  
  # Then save the best model.
  save_model_tf(quant.model, paste0("runs/QR_est/qr_fit"))
  
}

# Load in the quantile regression model 
quant.model <- tf$saved_model$load(paste0("runs/QR_est/qr_fit"))

# Predict radial quantiles for all angles  
pred.quant <- k_get_value(quant.model(k_constant(W)))

# Quick empirical check to assess suitability of estimated threshold function. Should be about the same 
print(mean(R <= pred.quant))
print(quant.level)

# Compute threshold exceedances, and create training, test and validation sets for each component
u <- pred.quant

R.train <- (R - u)[-valid.inds]
W.train <- W[-valid.inds, ]
u.train <- u[-valid.inds]
R.valid <- (R - u)[valid.inds]
W.valid <- W[valid.inds, ]
u.valid <- u[valid.inds]
R.test <- (R - u)[test.inds]
W.test <- W[test.inds, ]
u.test <- u[test.inds]

W.train <- W.train[R.train > 0, ]
u.train <- u.train[R.train > 0]
R.train <- R.train[R.train > 0]
W.valid <- W.valid[R.valid > 0, ]
u.valid <- u.valid[R.valid > 0]
R.valid <- R.valid[R.valid > 0]
W.test<- W.test[R.test > 0, ]
u.test <- u.test[R.test > 0]
R.test <- R.test[R.test > 0]

# Convert to matrices 
dim(R.train) = c(length(R.train), 1)
dim(R.valid) = c(length(R.valid), 1)
dim(R.test) = c(length(R.test), 1)

dim(u.train) = c(length(u.train), 1)
dim(u.valid) = c(length(u.valid), 1)
dim(u.test) = c(length(u.test), 1)

# The code for the GPD fitting is very similar to that for the quantile regression model. 
# Therefore, it is only partially commented

# Only fit model if flag is set to T
if(run_GPD){
  # Specify initial positive constant shape parameter for GPD network. 
  init_shape <- 0.05; if(init_shape>=0.1 | init_shape<0){stop("Initial shape be strictly between 0 and 0.1 ")}
  
  input.pseudo.angles <- layer_input(shape = d, name = "input.pseudo.angles")
  
  input.u <- layer_input(shape = d, name = "input.u")
  
  gpd.Branch <- input.pseudo.angles %>%
    layer_dense(units = gpd.nunits[1], activation = "relu", name = "gpd_dense1", kernel_regularizer = regularizer_l1_l2(l1 = 1e-04,
                                                                                                                        l2 = 1e-04))  #First hidden layer
  if (length(gpd.nunits) >= 2) {
    for (i in 2:length(gpd.nunits)) {
      gpd.Branch <- gpd.Branch %>%
        layer_dense(units = gpd.nunits[i], activation = "relu", name = paste0("gpd_dense", i), kernel_regularizer = regularizer_l1_l2(l1 = 1e-04,
                                                                                                                                      l2 = 1e-04))  #Subsequent hidden layers
    }
  }
  
  # Specify custom activation functions to ensure estimated shape is between (-0.5,0.1)
  GPD_custom_activation <- function(x) {
    tf$concat(c(activation_exponential(x[all_dims(),1:1]),
                0.3 * activation_tanh(x[all_dims(),2:2]) - 0.2),
              axis = 1L)
  }
  
  # Construct final layer such that initial output gives a constant shape parameter of init_shape
  gpd.Branch <- gpd.Branch %>%
    layer_dense(
      units = 2,
      activation = GPD_custom_activation,
      name = "gpd_final",
      weights = list(matrix(0, nrow = gpd.nunits[length(gpd.nunits)], ncol = 2), array(atanh((0.2 + init_shape) /
                                                                                               0.3
      ), dim = c(2))),
      kernel_regularizer = regularizer_l1_l2(l1 = 1e-04, l2 = 1e-04)
    )
  output <- layer_concatenate(c(gpd.Branch, input.u, input.pseudo.angles))
  
  GPD.model <- keras_model(inputs = c(input.pseudo.angles, input.u), outputs = output)
  summary(GPD.model)
  
  # Compile the model with the GPD loss and the adam optimiser 
  GPD.model %>%
    compile(optimizer = optimizer_adam(learning_rate = 0.001), loss = GPD_loss(0, penalty = 0), run_eagerly = T)
  
  
  # After every epoch, we use a checkpoint to save the weights.  Only the current best version of
  # the model is saved, i.e., the one that minimises the loss evaluated on the validation data
  
  checkpoint <- callback_model_checkpoint(filepath = paste0("runs/GPD_est/gpd_fit_{epoch:02d}"), 
                                          monitor = "val_loss", verbose = 0, save_best_only = FALSE, 
                                          save_weights_only = TRUE, mode = "min",save_freq = "epoch")
  
  n.epochs <- 500  
  
  # Given by definition there are few exceedances, we can set batch size to the entire data set
  batch.size = length(R.train)

  history <- GPD.model %>%
    fit(list(W.train, u.train), R.train, epochs = n.epochs, batch_size = batch.size, 
        callback = list(checkpoint,callback_early_stopping(monitor = "val_loss", min_delta = 0, patience = 10)), 
        validation_data = list(list(input.pseudo.angles = W.valid,input.u = u.valid), R.valid))
  
  # Select the epoch with the minimal loss function
  optimal_epoch = which.min(history$metrics$val_loss)
  
  # Load in the corresponding weights of the optimal epoch 

  GPD.model <- load_model_weights_tf(GPD.model,filepath = paste0("runs/GPD_est/gpd_fit_",sprintf("%02d", optimal_epoch)))
 
  # Save weights and model at optimal epoch
  save_model_weights_tf(GPD.model,  paste0("runs/GPD_est/gpd_fit_best_weights"))
  save_model_tf(GPD.model, paste0("runs/GPD_est/gpd_fit"))
  

  # Best Validation loss
  pred.GPD.valid <- k_get_value(GPD.model(list(k_constant(W.valid), k_constant(u.valid))))
  best.valid.loss <- k_get_value(GPD_loss(0, 0)(k_constant(R.valid), k_constant(pred.GPD.valid)))
  
  
  # The following code iteratively decreases the learning rate to see if we can improve the parameter estimates
  for (k in 1:length(l.r.vec)) {
    print(paste0("Decreasing learning rate size to: ", l.r.vec[k]))
    
    # Take the existing GPD model and continue optimising with a smaller learning rate 
    GPD.model %>%
      compile(optimizer = optimizer_adam(learning_rate = l.r.vec[k]), loss = GPD_loss(0, penalty = 0),
              run_eagerly = T)
    
    history <- GPD.model %>%
      fit(list(W.train, u.train), R.train, epochs = n.epochs, batch_size = batch.size, 
          callback = list(checkpoint,callback_early_stopping(monitor = "val_loss", min_delta = 0, patience = 10)), 
          validation_data = list(list(input.pseudo.angles = W.valid,input.u = u.valid), R.valid))
    
    # Check to see if validation loss has decreased. If so, save optimal epoch and loss value 
    if (length(which.min(history$metrics$val_loss)) > 0) {
      if (min(history$metrics$val_loss, na.rm = T) > best.valid.loss) {
        optimal_epoch = NA
        print("Validation loss has not decreased :(")
      } else {
        optimal_epoch = which.min(history$metrics$val_loss)

        print("Validation loss has decreased :)")
      }
    } else {
      optimal_epoch = NA
      print("Validation loss has not decreased :(")
    }
    
    # If we have improved loss function, we update the saved weights and model 
    if (!is.na(optimal_epoch)) {
      GPD.model <- load_model_weights_tf(
        GPD.model,
        filepath = paste0(
          "runs/GPD_est/gpd_fit_",
          sprintf("%02d", optimal_epoch))
      )
    
      
    } else {
    # Else, we revert back to previous weights 
      load_model_weights_tf(
        GPD.model,
        filepath = paste0(
          "runs/GPD_est/gpd_fit_best_weights"
        )
      )
    }
    
  }
  # Then save the best model.
  save_model_weights_tf(GPD.model, "runs/GPD_est/gpd_fit_best_weights")
  save_model_tf(GPD.model, paste0("runs/GPD_est/gpd_fit"))
  

  pred.GPD.valid <- k_get_value(GPD.model(list(k_constant(W.valid), k_constant(u.valid))))
  best.valid.loss <- k_get_value(GPD_loss(0, 0)(k_constant(R.valid), k_constant(pred.GPD.valid)))
  
  
}

# Load in GPD saved model 
GPD.model<- load_model_tf( "runs/GPD_est/gpd_fit",
                           custom_objects = list("GPD_loss_0__penalty___0_"=GPD_loss(0,0)))

# Validation --------------------------------------------------------------

# Compute GPD parameters over the test set 
pred.GPD.test <- k_get_value(GPD.model(list(k_constant(W.test), k_constant(u.test))))

# Use the probability integral transform to transform observed quantiles onto a standard exponential scale 
obs_quants = qexp(apply(cbind(u.test, pred.GPD.test[,1:2], R.test+u.test), 1, 
                        function(x) pgpd(x[4], loc = x[1], scale = x[2]/(1 +x[3]), x[3])))

# Compute length of observed quantiles 
n_p = length(R.test) 

# Compute corresponding theoretical quantiles on an exponential scale 
ps = (1:n_p)/(n_p + 1) 
theor_quants = qexp(ps)

# Save QQ plot for the GPD fit 
pdf(file=paste0("Validation/gpd_qq_plot.pdf"),width=5,height=5)

plot(theor_quants,sort(obs_quants),pch=16,col="grey",cex.lab=1.3, cex.axis=1.2,cex.main=1.5,xlim=range(sort(obs_quants),theor_quants),ylim=range(sort(obs_quants),theor_quants),xlab = "Theoretical",ylab = "Observed",main="GPD QQ plot")
abline(a=0,b=1,lwd=4,col=2)
points(theor_quants,sort(obs_quants),pch=16,col="black",cex=1.3)

dev.off()

# Probabilities at which to evaluate return level sets
probs = exp(seq(log(0.905),log(0.999),length.out=100))

# Corresponding probabilities for the GPD model 
trunc_probs = (probs-quant.level)/(1-quant.level)

# Function for evaluating empirical probability of radial data lying inside return level set 
empirical_probs_function = function(tp){
  # Compute radial quantiles on return level set for each probability  
  radial_quants = apply(cbind(tp,u.test,pred.GPD.test[,1:2]),1,function(x)
    qgpd(x[1], loc=x[2], scale = x[3]/(1+ x[4]), shape=x[4])
  )
  # Compute proportion of points inside return level set 
  return(mean(R.test+u.test<=radial_quants))
  
}

# Compute empirical probabilities of being inside each return level set 
empirical_probs = sapply(trunc_probs,empirical_probs_function)

# Transform back to original scale  
empirical_probs = empirical_probs*(1-quant.level) + quant.level

pdf(file=paste0("Validation/ret_level_set_probs.pdf"),width=5,height=5)

#Plotting parameters
par(mfrow=c(1,1),mgp=c(2.25,0.75,0),mar=c(4,4,1,1))

plot(-log(1-probs),-log(1-empirical_probs),xlim=range(-log(1-probs),-log(1-empirical_probs)),
     ylim=range(-log(1-probs),-log(1-empirical_probs)),pch=16,col="grey",
     xlab = "Theoretical",ylab = "Observed",sub=expression(-log(1-p)),cex.lab=1.3, cex.axis=1.2,cex.main=1.8, cex=0.5)
abline(a=0,b=1,lwd=3,col=2)
points(-log(1-probs),-log(1-empirical_probs),pch=16,col=1, cex=0.8)

dev.off()

# Finally, we simulate data from the fitted model and compare with the observations  

# Sepcify number of simulation 
n.sim <- 1e7

# Simulate uniform observation 
U <- runif(n.sim,0,1)

# Subset normalised data by points in and outside of the quantile set
X.norm.nonexcess = X.norm[which(pred.quant > R),]
X.norm.excess = X.norm[which(pred.quant <= R),]

# Specify how many observation to simulation from the body (in the quantile set)
num.emp <- sum(U <= quant.level)

# Sample data in the body empirically 
X.sim = X.norm.nonexcess[sample(1:nrow(X.norm.nonexcess), num.emp,replace=T),]

# Sample angles with replacement 
W.sim = W[sample(1:nrow(W), n.sim-num.emp,replace=T),]

# Compute threshold at sample angles 
u.sim <- k_get_value(quant.model(k_constant(W.sim)))

# Convert to matrix 
dim(u.sim) = c(length(u.sim), 1)

# Find GPD parameters at sample angles 
pred.GPD.sim <- k_get_value(GPD.model(list(k_constant(W.sim), k_constant(u.sim))))

# Extract GPD parameters 
nu.sim = pred.GPD.sim[, 1]
xi.sim = pred.GPD.sim[, 2]

# Convert uniform quantiles to quantiles on the threshold exceedance scale 
unif_sample = (U[U>quant.level]-quant.level)/(1-quant.level)

# Simulate radial observations from GPD
R.sim = apply(cbind(unif_sample,u.sim,nu.sim,xi.sim),1,function(x)
  qgpd(x[1], loc=x[2], scale = x[3]/(1+ x[4]), shape=x[4])
)

# Combined non-extreme and extreme simulations 
X.sim = rbind(X.sim, R.sim*W.sim) 

# Assess models ability to capture marginal tails 
pdf(file=paste0("Validation/marginal_qq_plots.pdf"),width=15,height=6)

par(mfrow=c(2,5))

for(i in 1:d){
  
  title = wave.names[i]
  
  # Specify quantiles in the upper tail of the marginal variables 
  ps  = seq(0.99, 1 - 1/nrow(X.norm), length = 500)
  
  # Compute observed and model quantiles 
  obs_quants = quantile(X.norm[,i],ps)
  sim_quants = quantile(X.sim[,i],ps)
  
  # Rescaled all quantiles to lay in the interval [0,1]
  max_quant = max(c(obs_quants,sim_quants))
  min_quant = min(c(obs_quants,sim_quants))
  
  obs_quants = (obs_quants - min_quant)/(max_quant - min_quant)
  sim_quants = (sim_quants - min_quant)/(max_quant - min_quant)
  
  # Plot the observed and model quantiles 
  plot(sim_quants,obs_quants,pch=16,col="grey",cex.lab=1.3, cex.axis=1.2,cex.main=1.5,xlim=range(obs_quants,sim_quants),ylim=range(obs_quants,sim_quants),ylab = "Observed",xlab = "Model",main=title,sub="Upper tail (scaled)")
  abline(a=0,b=1,lwd=4,col=2)
  points(sim_quants,obs_quants,pch=16,col="black",cex=1.3)
  
  
}


for(i in 1:d){
  
  title = wave.names[i]
  
  # Specify quantiles in the lower tail of the marginal variables 
  ps  = seq( 1/nrow(X.norm),0.01, length = 500)
  
  obs_quants = quantile(X.norm[,i],ps)
  sim_quants = quantile(X.sim[,i],ps)
  
  max_quant = max(c(obs_quants,sim_quants))
  min_quant = min(c(obs_quants,sim_quants))
  
  obs_quants = (max_quant - obs_quants)/(max_quant - min_quant)
  sim_quants = (max_quant - sim_quants)/(max_quant - min_quant)
  
  plot(sim_quants,obs_quants,pch=16,col="grey",cex.lab=1.3, cex.axis=1.2,cex.main=1.5,xlim=range(obs_quants,sim_quants),ylim=range(obs_quants,sim_quants),ylab = "Observed",xlab = "Model",main=title,sub="Lower tail (scaled)")
  abline(a=0,b=1,lwd=4,col=2)
  points(sim_quants,obs_quants,pch=16,col="black",cex=1.3)
  
  
}

dev.off()

# Transform all observed and model extreme observations back to the original scale 
X.excess.obs = apply(rbind(means,sds,X.norm.excess),2,function(x){return(x[-c(1:2)]*x[2] + x[1])})

X.excess.sim = apply(rbind(means,sds,(R.sim*W.sim)),2,function(x){return(x[-c(1:2)]*x[2] + x[1])})

png(file=paste0("Validation/scatterplot.png"),width=800,height=800,res = 100)

labels <- wave.names

pairs(rbind(X.excess.sim,X.excess.obs),labels = labels,main="Generated vs observed extremes",pch=16,col=c(rep(rgb(0, 1, 0, alpha = 0.3),nrow(X.excess.sim)),rep("grey",nrow(X.excess.obs))),cex.labels=1.3)

dev.off()

