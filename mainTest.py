url = "https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c"

from adam_class import Adam
from matplotlib.animation import PillowWriter

f = lambda x : x[0]**2 + 4*x[1]**2
th0 =[3,4.5]
adam_optimizer = Adam(f, th0, lr = 0.01)
x = adam_optimizer.fit()
adam_optimizer.summary()
adam_optimizer.plot_gradient_norms()

x = [-5,5]
parameter = dict(minimum = [0,0], x_lim = x, y_lim = x, label = "Adam", n_con= 100,  sign= '-', c='b')
plot_g = adam_optimizer.plot_g(**parameter)
plot_g2 = adam_optimizer.plot_g2(sc =0.02, **parameter)
plot_m = adam_optimizer.plot_m(sc =0.5, **parameter)
plot_v = adam_optimizer.plot_v(sc = 0.02, **parameter)
plot_m_g = adam_optimizer.plot_m_g(sc=0.5, **parameter)
plot_v_g2 = adam_optimizer.plot_v_g2(sc=0.02, **parameter)
plot_d = adam_optimizer.plot_d(sc =100, **parameter)

files_path = ['plot_g.gif', 'plot_g2.gif', 'plot_m.gif', 'plot_v.gif',
               'plot_m_g.gif', 'plot_v_g2.gif', 'plot_d.gif']
anims = [plot_g, plot_g2, plot_m, plot_v, plot_m_g, plot_v_g2, plot_d]
for anim, file in zip(anims, files_path):
    anim.save(file, writer=PillowWriter(fps=30))