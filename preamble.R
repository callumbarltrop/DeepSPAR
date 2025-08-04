
rm(list=ls())

# Functions/packages ------------------------------------------------------
# For quantile estimation, we use the check/pinball/tilted loss.
tilted_loss <- function(y_true, y_pred) {
  K <- backend()
  
  error = y_true - y_pred
  return(K$mean(K$maximum(quant.level * error, (quant.level - 1) * error)))
}

rect2polar <- function (x) 
{
  if (!is.matrix(x)) {
    x <- as.matrix(x, ncol = 1)
  }
  n <- nrow(x)
  m <- ncol(x)
  r <- rep(0, m)
  phi <- matrix(0, nrow = n - 1, ncol = m)
  for (j in 1:m) {
    y <- sqrt(cumsum(rev(x[, j]^2)))
    r[j] <- y[n]
    if (r[j] > 0) {
      if (n > 2) {
        for (k in 1:(n - 2)) {
          if (y[n - k + 1] > 0) 
            phi[k, j] <- acos(x[k, j]/y[n - k + 1])
          else {
            phi[k, j] <- ifelse(x[k, j] > 0, 0, pi)
          }
        }
      }
      if (y[2] > 0) {
        phi[n - 1, j] <- acos(x[n - 1, j]/y[2])
        if (x[n, j] < 0) {
          phi[n - 1, j] <- 2 * pi - phi[n - 1, j]
        }
      }
      else {
        phi[n - 1, j] <- ifelse(x[n, j] > 0, 0, pi)
      }
    }
  }
  return(list(r = r, phi = phi))
}

polar2rect <- function (r, phi) 
{
  m <- length(r)
  if (!is.matrix(phi)) {
    phi <- as.matrix(phi, ncol = 1)
  }
  stopifnot(m == ncol(phi))
  n <- nrow(phi) + 1
  x <- matrix(0, nrow = n, ncol = m)
  for (j in 1:m) {
    c.term <- cos(phi[, j])
    s.term <- sin(phi[, j])
    y <- c(1, cumprod(s.term))
    z <- c(c.term, 1)
    x[, j] <- r[j] * y * z
  }
  return(x)
}


GPD_loss <- function(T0,penalty){
  
  loss<-function( y_true, y_pred) {
    
    K <- backend()
    
    nu=y_pred[all_dims(),1] #See Pasche and Engelke, 2023
    xi=y_pred[all_dims(),2]
    u=y_pred[all_dims(),3]
    y=y_true[all_dims(),1]
 
    w_T=y_pred[all_dims(),ncol(y_pred)]

    #Evaluate log-likelihood
    ll1=(1/xi+1)*tf$math$log1p(xi*(xi+1)*y/nu)

    ll2= K$log(nu) -K$log(1+xi)
    
    w_T_neg_index=K$sign(K$relu(-w_T)) 
    xi_neg_index=K$sign(K$relu(-xi)) 
    
    Rmax = u - nu/(xi*(xi+1))
    
    ll3 = (K$relu(T0-Rmax *w_T) *w_T_neg_index*xi_neg_index)^2
    
    return((K$mean(ll1+ll2))+penalty * K$mean(ll3) )
 
}
}

#Checking for required packages. This function will install any required packages if they are not already installed
packages = c("ismev","evd","mvtnorm") 
package.check <- lapply(
  packages,
  FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
      install.packages(x, dependencies = TRUE)
      library(x, character.only = TRUE)
    }
  }
)

block_bootstrap_function = function(data,k,n=length(as.matrix(data)[,1])){ #function for performing block bootstrapping
  #data is bivariate dataset
  #k is block length
  data = as.matrix(data)
  no_blocks = ceiling(n/k)
  n_new = no_blocks*k
  new_data = matrix(NA,nrow=n_new,ncol=dim(data)[2])
  indices = 1:(n-k+1)
  start_points = sample(x=indices,size=no_blocks,replace=TRUE)
  for(i in 1:no_blocks){
    new_data[((i-1)*k+1):(i*k),] = data[(start_points[i]:(start_points[i]+k-1)),]
  }
  return(new_data[1:n,])
}