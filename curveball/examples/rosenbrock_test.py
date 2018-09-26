
import tensorflow as tf
import curveball
#from tf.train import GradientDescentOptimizer

def noisy_rosenbrock(u, v, noise_range=0):
    noise = tf.random_uniform([1], 0, 1) * noise_range + 1
    return (1-u) ** 2 + 100 * noise * (v - u ** 2) ** 2

print(noisy_rosenbrock(1,1))

def optimization_loop(optimizer, loss, train_step):

    #loss = noisy_rosenbrock(u, v)
    init = tf.initialize_all_variables()
    with tf.train.MonitoredTrainingSession() as sess:
        sess.run(init)
        for i in range(5):
        #while not sess.should_stop():
        #    sess.close()

            [loss_, _] = sess.run([loss, train_step])
            print(loss_)
            print("        ")
          #global_step_, loss_, accuracy_, _ = sess.run(
          #    [g_step, loss, accuracy, train_op])

            #if i % 1 == 0:
            #    tf.logging.info("global_step: %d | loss: %f | accuracy: %s",
            #        global_step_, loss_, accuracy_)



gd_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001,
                name='GradientDescent')


u = tf.get_variable("u", dtype=tf.float32, initializer=tf.constant([0.25]))
v = tf.get_variable("v", dtype=tf.float32, initializer=tf.constant([0.25]))#, trainable=False)
uv = tf.concat([u, v], axis=0)
print(uv)
loss = noisy_rosenbrock(uv[0], uv[1])
init = tf.initialize_all_variables()
#with tf.train.MonitoredTrainingSession() as sess:
#    sess.run(init)
#    feed = {u:u, v:v}
#    loss_ = sess.run(loss)
#    print(loss)

#train_step = gd_optimizer.minimize(loss)
#optimization_loop(gd_optimizer, loss, train_step)

Hl = tf.hessians(loss, uv)
print("Hl function:", Hl)
with tf.Session() as sess:
    sess.run(init)
    Hl_ = sess.run(Hl)
    print(Hl_)
    

# Curveball Optimizer
curveball_optimizer = curveball.CurveballOptimizer(learning_rate=1, name='Curveball', input_to_loss=uv)
train_step = curveball_optimizer.minimize(loss)
optimization_loop(curveball_optimizer, loss, train_step)



