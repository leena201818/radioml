import tensorflow as tf
from math import pi

tf.enable_eager_execution()
tfe = tf.contrib.eager # Shorthand for some symbols

def f(x):
  return tf.square(tf.sin(x))

assert f(pi/2).numpy() == 1.0

# grad_f will return a list of derivatives of f
# with respect to its arguments. Since f() has a single argument,
# grad_f will return a list with a single element.
grad_f = tfe.gradients_function(f)
print( grad_f(pi/4)[0] )

print( tf.abs(grad_f(pi/4))[0].numpy() )
assert tf.abs(grad_f(pi/2)[0]).numpy() < 1e-7
