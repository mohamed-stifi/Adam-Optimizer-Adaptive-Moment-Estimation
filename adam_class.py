from numdifftools import Gradient
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
class Adam :
    def __init__(self, f, th0, lr = 1e-3, b1 = 0.9, b2 = 0.99, eps = 1e-8, tol = 1e-8) -> None:
        self.fun = f
        self.Gf = Gradient(f)
        self.x = [th0]
        self.alpha = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.tol = tol
        self.g = [self.Gf(self.x[-1])]
        self.m = [0]
        self.v = [0]
        self.m_hat = [0]
        self.v_hat = [0]
        self.t = 0
        self.b1_pow_t = 1
        self.b2_pow_t = 1

    def fit(self, max_t = 1000):
        self.max_t = max_t
        while np.linalg.norm(self.g[-1]) > self.tol and self.t < self.max_t :
            self.t += 1
            self.m.append(self.b1 * self.m[-1] + (1 - self.b1)*self.g[-1])
            self.v.append(self.b2 * self.v[-1] + (1 - self.b2)*self.g[-1]**2)
            self.b1_pow_t *= self.b1
            self.b2_pow_t *= self.b2
            self.m_hat.append(self.m[-1]/(1-self.b1_pow_t))
            self.v_hat.append(self.v[-1]/(1-self.b2_pow_t))
            self.x.append(self.x[-1] - self.alpha * self.m_hat[-1] /(np.sqrt(self.v_hat[-1]) + self.eps))
        
            self.g.append(self.Gf(self.x[-1]))
            
        
        return self.x[-1]

    def plot_gradient_norms(self):
        gradient_norms = [np.linalg.norm(g) for g in self.g]
        plt.plot(gradient_norms, label='Gradient Norms')
        plt.xlabel('Iterations')
        plt.ylabel('Gradient Norms')
        plt.title('Adam Optimizer - Gradient Norms')
        plt.legend()
        plt.show()
        
    
    def summary(self):
        print("Adam Optimizer Summary:")
        print(f"Initial Parameters: x0 = {self.x[0]}, learning rate = {self.alpha}, b1 = {self.b1}, b2 = {self.b2}")
        print(f"Converged Parameters: x_final = {self.x[-1]}")
        print(f"Number of Iterations: {self.t}/{self.max_t}")
        print(f"Final Gradient Norm: {np.linalg.norm(self.g[-1])}")
        print(f"Convergence Tolerance: {self.tol}")
    
    def plot_distance_from_minimum(self, true_minimum):
        distances = [np.linalg.norm(x - true_minimum) for x in self.x]
        plt.plot(distances, label='$\\|x_t - min f(x)\\|$')
        plt.xlabel('Iterations')
        plt.ylabel('Distance from Actual Minimum')
        plt.title('Adam Optimizer - Distance from Actual Minimum')
        plt.legend()
        plt.show()
        

    def plot_x(self,minimum, x_lim, y_lim, label, step = 10, n_seconds =5,
                     c= 'r', n_points = 300, n_con =100, figsize = (14,16)):
        """
    Visualizes the optimization path on a contour plot of the objective function.

    Parameters:
    - minimum: Tuple, coordinates of the true minimum of the objective function.
    - x_lim: Tuple, limits for the x-axis in the plot.
    - y_lim: Tuple, limits for the y-axis in the plot.
    - label: String, label for the animation in the legend.
    - step: Integer, step size for sampling the optimization path.
    - n_seconds: Integer, total animation duration in seconds.
    - c: String, color for the animation path.
    - n_points: Integer, number of points along each axis for contour plot.
    - n_con: Integer, number of contour lines in the plot.
    - figsize: Tuple, size of the figure for the plot.

    Returns:
    - anim: Animation object showing the optimization path.
    """
        try :
            path_length = int(self.t/step)
            path = np.array(self.x[:-step:step])
            #path = np.concatenate((path,np.array(self.x[-step:])), axis = 0)
            x = np.linspace(*x_lim, n_points)
            y = np.linspace(*y_lim, n_points)
            X, Y = np.meshgrid(x, y)
            Z = self.fun([X,Y])
            fig, ax = plt.subplots(figsize=figsize)
            ax.contour(X, Y, Z, n_con, cmap="jet")
            ax.scatter(*minimum, c = 'black', label = 'min')
            scatter = ax.scatter(None,
                                None,
                                label= label,
                                c=c) 
            
            ax.legend(prop={"size": 25})
            #ax.plot(optimal_points[:,0], optimal_points[:,1], colors)
            
            def animate(i):
                
                scatter.set_offsets(path[:i, :])
                
                ax.set_title(str(step*i ))
            ms_per_frame = 1000 * n_seconds / path_length
            anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)
            plt.show()
        except Exception as e :
            print(e)
            return
        return anim

    def plot_g(self,minimum, x_lim, y_lim, label, sign = '',step = 10, n_seconds =5,
                    sc=1, c= 'r', n_points = 300, n_con =100, figsize = (14,16)):
        """
    Visualizes the optimization path and gradient vectors on a contour plot of the objective function.

    Parameters:
    - minimum: Tuple, coordinates of the true minimum of the objective function.
    - x_lim: Tuple, limits for the x-axis in the plot.
    - y_lim: Tuple, limits for the y-axis in the plot.
    - label: String, label for the animation in the legend.
    - sign: String, sign to indicate positive or negative gradient vectors.
    - step: Integer, step size for sampling the optimization path.
    - n_seconds: Integer, total animation duration in seconds.
    - sc: Integer, scaling factor for the gradient vectors.
    - c: String, color for the animation path.
    - n_points: Integer, number of points along each axis for contour plot.
    - n_con: Integer, number of contour lines in the plot.
    - figsize: Tuple, size of the figure for the plot.

    Returns:
    - anim: Animation object showing the optimization path and gradient vectors.
    """
        try :
            path_length = int(self.t/step)
            path = np.array(self.x[::step])
            g = np.array(self.g[::step])
            #path = np.concatenate((path,np.array(self.x[-step:])), axis = 0)
            x = np.linspace(*x_lim, n_points)
            y = np.linspace(*y_lim, n_points)
            X, Y = np.meshgrid(x, y)
            Z = self.fun([X,Y])
            fig, ax = plt.subplots(figsize=figsize)
            ax.contour(X, Y, Z, n_con, cmap='viridis')
            ax.scatter(*minimum, c = 'black', label = 'min')
            scatter = ax.scatter(None,
                                None,
                                label= label,
                                c=c) 
            
            quiver = ax.quiver(0, 0, 0, 0, width=0.005, linewidth=0.5,
                               angles='xy', scale_units='xy', scale=5, 
                               color='r', label=str(sign)+'g[t]')
            
            ax.legend(prop={"size": 25})
            #ax.plot(optimal_points[:,0], optimal_points[:,1], colors)
            s = -1 if sign == '-' else 1
            
            def animate(i):
                current_x = path[i]
                current_g = sc*g[i]
                
                scatter.set_offsets(path[:i, :])
                quiver.set_offsets(np.array([current_x[0], current_x[1]]))
                quiver.set_UVC(s*current_g[0], s*current_g[1])

                title = f"i: {i} / {self.t}/{step} | g{i} = {current_g}/{sc}."
                ax.set_title(title)
            ms_per_frame = 1000 * n_seconds / path_length
            anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)
            plt.show()
        except Exception as e :
            print(e)
            return
        return anim

    def plot_m(self,minimum, x_lim, y_lim, label,sign = '', step = 10, n_seconds =5,
                    sc=1, c= 'r', n_points = 300, n_con =100, figsize = (14,16)):
        """
    Visualizes the optimization path and first momentum vectors on a contour plot of the objective function.

    Parameters:
    - minimum: Tuple, coordinates of the true minimum of the objective function.
    - x_lim: Tuple, limits for the x-axis in the plot.
    - y_lim: Tuple, limits for the y-axis in the plot.
    - label: String, label for the animation in the legend.
    - sign: String, sign to indicate positive or negative momentum vectors.
    - step: Integer, step size for sampling the optimization path.
    - n_seconds: Integer, total animation duration in seconds.
    - sc: Integer, scaling factor for the momentum vectors.
    - c: String, color for the animation path.
    - n_points: Integer, number of points along each axis for contour plot.
    - n_con: Integer, number of contour lines in the plot.
    - figsize: Tuple, size of the figure for the plot.

    Returns:
    - anim: Animation object showing the optimization path and momentum vectors.
    """
        try :
            path_length = int(self.t/step)
            path = np.array(self.x[::step])
            m = np.array(self.m[1::step])

            x = np.linspace(*x_lim, n_points)
            y = np.linspace(*y_lim, n_points)
            X, Y = np.meshgrid(x, y)
            Z = self.fun([X,Y])
            fig, ax = plt.subplots(figsize=figsize)
            ax.contour(X, Y, Z, n_con, cmap='viridis')
            ax.scatter(*minimum, c = 'black', label = 'min')
            scatter = ax.scatter(None,
                                None,
                                label= label,
                                c=c) 
            
            quiver = ax.quiver(0, 0, 0, 0, width=0.005, linewidth=0.5,
                               angles='xy', scale_units='xy', scale=5, 
                               color='r', label=sign + 'm[t]')
            
            ax.legend(prop={"size": 25})

            s = -1 if sign == '-' else 1
            def animate(i):
                current_x = path[i]
                current_m = sc*m[i]
                
                scatter.set_offsets(path[:i, :])
                quiver.set_offsets(np.array([current_x[0], current_x[1]]))
                quiver.set_UVC(s*current_m[0], s*current_m[1])

                title = f"i: {i} / {self.t}/{step} | m{i} = {current_m}/{sc}."
                ax.set_title(title)
            ms_per_frame = 1000 * n_seconds / path_length
            anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)
            plt.show()
        except Exception as e :
            print(e)
            return
        return anim

    def plot_v(self,minimum, x_lim, y_lim, label, sign = '',step = 10, n_seconds =5,
                    sc=1, c= 'r', n_points = 300, n_con =100, figsize = (14,16)):
        """
    Visualizes the optimization path and velocity vectors on a contour plot of the objective function.

    Parameters:
    - minimum: Tuple, coordinates of the true minimum of the objective function.
    - x_lim: Tuple, limits for the x-axis in the plot.
    - y_lim: Tuple, limits for the y-axis in the plot.
    - label: String, label for the animation in the legend.
    - sign: String, sign to indicate positive or negative velocity vectors.
    - step: Integer, step size for sampling the optimization path.
    - n_seconds: Integer, total animation duration in seconds.
    - sc: Integer, scaling factor for the velocity vectors.
    - c: String, color for the animation path.
    - n_points: Integer, number of points along each axis for contour plot.
    - n_con: Integer, number of contour lines in the plot.
    - figsize: Tuple, size of the figure for the plot.

    Returns:
    - anim: Animation object showing the optimization path and velocity vectors.
    """
        try :
            path_length = int(self.t/step)
            path = np.array(self.x[::step])
            v = np.array(self.v[1::step])

            x = np.linspace(*x_lim, n_points)
            y = np.linspace(*y_lim, n_points)
            X, Y = np.meshgrid(x, y)
            Z = self.fun([X,Y])

            fig, ax = plt.subplots(figsize=figsize)
            ax.contour(X, Y, Z, n_con, cmap='viridis')
            ax.scatter(*minimum, c = 'black', label = 'min')
            scatter = ax.scatter(None,
                                None,
                                label= label,
                                c=c) 
            
            quiver = ax.quiver(0, 0, 0, 0, width=0.005, linewidth=0.5,
                               angles='xy', scale_units='xy', scale=5,
                               color='r', label=sign+'v[t]')
            
            ax.legend(prop={"size": 25})
            
            s = -1 if sign == '-' else 1

            def animate(i):
                current_x = path[i]
                current_v = sc*v[i]
                
                scatter.set_offsets(path[:i, :])
                quiver.set_offsets(np.array([current_x[0], current_x[1]]))
                quiver.set_UVC(s*current_v[0], s*current_v[1])

                title = f"i: {i} / {self.t}/{step} | v{i} = {current_v}/{sc}"
                ax.set_title(title)
            ms_per_frame = 1000 * n_seconds / path_length
            anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)
            plt.show()
        except Exception as e :
            print(e)
            return
        return anim

    def plot_m_g(self,minimum, x_lim, y_lim, label,sign='', step = 10, n_seconds =5,
                   sc =1,  c= 'r', n_points = 300, n_con =100, figsize = (14,16)):
        """
    Visualizes the optimization path, gradient vectors, and momentum vectors on a contour plot of the objective function.

    Parameters:
    - minimum: Tuple, coordinates of the true minimum of the objective function.
    - x_lim: Tuple, limits for the x-axis in the plot.
    - y_lim: Tuple, limits for the y-axis in the plot.
    - label: String, label for the animation in the legend.
    - sign: String, sign to indicate positive or negative vectors.
    - step: Integer, step size for sampling the optimization path.
    - n_seconds: Integer, total animation duration in seconds.
    - sc: Integer, scaling factor for the vectors.
    - c: String, color for the animation path.
    - n_points: Integer, number of points along each axis for contour plot.
    - n_con: Integer, number of contour lines in the plot.
    - figsize: Tuple, size of the figure for the plot.

    Returns:
    - anim: Animation object showing the optimization path, gradient vectors, and momentum vectors.
    """
        try :
            path_length = int(self.t/step)
            path = np.array(self.x[::step])
            g = (1-self.b1)*np.array(self.g[::step])
            m = np.array(self.m[1::step])

            x = np.linspace(*x_lim, n_points)
            y = np.linspace(*y_lim, n_points)
            X, Y = np.meshgrid(x, y)
            Z = self.fun([X,Y])

            fig, ax = plt.subplots(figsize=figsize)
            ax.contour(X, Y, Z, n_con, cmap='viridis')
            ax.scatter(*minimum, c = 'black', label = 'min')
            scatter = ax.scatter(None,
                                None,
                                label= label,
                                c=c) 
            
            quiver = ax.quiver(0, 0, 0, 0, width=0.005, linewidth=0.5,
                               angles='xy', scale_units='xy', scale=5, color='r', label=sign+'b1*g[t]')
            quiver1 = ax.quiver(0, 0, 0, 0, width=0.005, linewidth=0.5,
                               angles='xy', scale_units='xy', scale=5, color='y', label=sign+'(1- b1)*m[t-1]')
            quiver2 = ax.quiver(0, 0, 0, 0, width=0.005, linewidth=0.5,
                               angles='xy', scale_units='xy', scale=5, color='g', label=sign+'m[t]')
            
            ax.legend(prop={"size": 25})
            
            s = -1 if sign == '-' else 1

            def animate(i):
                if i > 0:
                    current_x = path[i]
                    current_g = sc*g[i]
                    m1 = sc*m[i-1]
                    current_m = sc*m[i]

                    scatter.set_offsets(path[:i, :])
                    quiver.set_offsets(np.array([current_x[0], current_x[1]]))
                    quiver.set_UVC(s*current_g[0], s*current_g[1])

                    quiver1.set_offsets(np.array([current_x[0], current_x[1]]))
                    quiver1.set_UVC(s*self.b1*m1[0], s*self.b2*m1[1])

                    quiver2.set_offsets(np.array([current_x[0], current_x[1]]))
                    quiver2.set_UVC(s*current_m[0], s*current_m[1])

                    title = f"i: {i} / {self.t}/{step} | g{i} = {current_g}/{sc} | m{i} = {current_m}/{sc} m{i-1} = {m1}/{sc}"
                    ax.set_title(title)
            ms_per_frame = 1000 * n_seconds / path_length
            anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)
            plt.show()
        except Exception as e :
            print(e)
            return
        return anim

    def plot_v_g2(self,minimum, x_lim, y_lim, label, sign ='', step = 10, n_seconds =5,
                   sc =1,  c= 'r', n_points = 300, n_con =100, figsize = (14,16)):
        """
    Visualizes the optimization path, gradient vectors squared, and second moment vectors on a contour plot of the objective function.

    Parameters:
    - minimum: Tuple, coordinates of the true minimum of the objective function.
    - x_lim: Tuple, limits for the x-axis in the plot.
    - y_lim: Tuple, limits for the y-axis in the plot.
    - label: String, label for the animation in the legend.
    - sign: String, sign to indicate positive or negative vectors.
    - step: Integer, step size for sampling the optimization path.
    - n_seconds: Integer, total animation duration in seconds.
    - sc: Integer, scaling factor for the vectors.
    - c: String, color for the animation path.
    - n_points: Integer, number of points along each axis for contour plot.
    - n_con: Integer, number of contour lines in the plot.
    - figsize: Tuple, size of the figure for the plot.

    Returns:
    - anim: Animation object showing the optimization path, gradient vectors squared, and second moment vectors.
    """
        try :
            path_length = int(self.t/step)
            path = np.array(self.x[::step])
            g = (1-self.b2)*np.array(self.g[::step])
            v = np.array(self.v[1::step])

            x = np.linspace(*x_lim, n_points)
            y = np.linspace(*y_lim, n_points)
            X, Y = np.meshgrid(x, y)
            Z = self.fun([X,Y])

            fig, ax = plt.subplots(figsize=figsize)
            ax.contour(X, Y, Z, n_con, cmap='viridis')
            ax.scatter(*minimum, c = 'black', label = 'min')
            scatter = ax.scatter(None,
                                None,
                                label= label,
                                c=c) 
            
            quiver = ax.quiver(0, 0, 0, 0, width=0.005, linewidth=0.5,
                               angles='xy', scale_units='xy', scale=5, color='r', label=sign+'b2*g[t]^2')
            quiver1 = ax.quiver(0, 0, 0, 0, width=0.005, linewidth=0.5,
                               angles='xy', scale_units='xy', scale=5, color='y', label=sign+'(1- b2)*v[t-1]')
            quiver2 = ax.quiver(0, 0, 0, 0, width=0.005, linewidth=0.5,
                               angles='xy', scale_units='xy', scale=5, color='g', label=sign+'v[t]')
            
            ax.legend(prop={"size": 25})
            #ax.plot(optimal_points[:,0], optimal_points[:,1], colors)
            s = -1 if sign == '-' else 1

            def animate(i):
                if i > 0:
                    current_x = path[i]
                    current_g = sc*g[i]
                    v1 = sc*v[i-1]
                    current_v = sc*v[i]
                    scatter.set_offsets(path[:i, :])
                    quiver.set_offsets(np.array([current_x[0], current_x[1]]))
                    quiver.set_UVC(s*current_g[0]**2, s*current_g[1]**2)

                    quiver1.set_offsets(np.array([current_x[0], current_x[1]]))
                    quiver1.set_UVC(s*self.b2*v1[0], s*self.b2*v1[1])

                    quiver2.set_offsets(np.array([current_x[0], current_x[1]]))
                    quiver2.set_UVC(s*current_v[0], s*current_v[1])

                    title = f"i: {i} / {self.t}/{step} | g{i} = {current_g}/{sc} | v{i} = {current_v}/{sc} v{i-1} = {v1}/{sc}"
                    ax.set_title(title)
            ms_per_frame = 1000 * n_seconds / path_length
            anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)
            plt.show()
        except Exception as e :
            print(e)
            return
        return anim

    def plot_g2(self,minimum, x_lim, y_lim, label, sign = '',step = 10, n_seconds =5,
                   sc =1,  c= 'r', n_points = 300, n_con =100, figsize = (14,16)):
        """
    Visualizes the optimization path and squared gradient vectors on a contour plot of the objective function.

    Parameters:
    - minimum: Tuple, coordinates of the true minimum of the objective function.
    - x_lim: Tuple, limits for the x-axis in the plot.
    - y_lim: Tuple, limits for the y-axis in the plot.
    - label: String, label for the animation in the legend.
    - sign: String, sign to indicate positive or negative vectors.
    - step: Integer, step size for sampling the optimization path.
    - n_seconds: Integer, total animation duration in seconds.
    - sc: Integer, scaling factor for the vectors.
    - c: String, color for the animation path.
    - n_points: Integer, number of points along each axis for contour plot.
    - n_con: Integer, number of contour lines in the plot.
    - figsize: Tuple, size of the figure for the plot.

    Returns:
    - anim: Animation object showing the optimization path and squared gradient vectors.
    """
        try :
            path_length = int(self.t/step)
            path = np.array(self.x[::step])
            g2 = np.array([g**2 for g in self.g[1::step]])

            x = np.linspace(*x_lim, n_points)
            y = np.linspace(*y_lim, n_points)
            X, Y = np.meshgrid(x, y)
            Z = self.fun([X,Y])

            fig, ax = plt.subplots(figsize=figsize)
            ax.contour(X, Y, Z, n_con, cmap='viridis')
            ax.scatter(*minimum, c = 'black', label = 'min')
            scatter = ax.scatter(None,
                                None,
                                label= label,
                                c=c) 
            
            quiver = ax.quiver(0, 0, 0, 0, width=0.005, linewidth=0.5,
                               angles='xy', scale_units='xy', scale=5,
                               color='r', label=sign+'g^2[t]')
            
            ax.legend(prop={"size": 25})
            
            s = -1 if sign == '-' else 1

            def animate(i):
                current_x = path[i]
                current_g2 = sc*g2[i]
                
                scatter.set_offsets(path[:i, :])
                quiver.set_offsets(np.array([current_x[0], current_x[1]]))
                quiver.set_UVC(s*current_g2[0], s*current_g2[1])

                title = f"i: {i} / {self.t}/{step} | $g^2${i} = {current_g2}/{sc}."
                ax.set_title(title)
            ms_per_frame = 1000 * n_seconds / path_length
            anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)
            plt.show()
        except Exception as e :
            print(e)
            return
        return anim

    def plot_d(self,minimum, x_lim, y_lim, label, sign = '',step = 10, n_seconds =5,
                    sc = 1, c= 'g', n_points = 300, n_con =100, figsize = (14,16)):
        """
    Visualizes the optimization path and update direction vectors on a contour plot of the objective function.

    Parameters:
    - minimum: Tuple, coordinates of the true minimum of the objective function.
    - x_lim: Tuple, limits for the x-axis in the plot.
    - y_lim: Tuple, limits for the y-axis in the plot.
    - label: String, label for the animation in the legend.
    - sign: String, sign to indicate positive or negative vectors.
    - step: Integer, step size for sampling the optimization path.
    - n_seconds: Integer, total animation duration in seconds.
    - sc: Integer, scaling factor for the vectors.
    - c: String, color for the animation path.
    - n_points: Integer, number of points along each axis for contour plot.
    - n_con: Integer, number of contour lines in the plot.
    - figsize: Tuple, size of the figure for the plot.

    Returns:
    - anim: Animation object showing the optimization path and update vectors d.
    """
        try :
            path_length = int(self.t/step)
            path = np.array(self.x[::step])
            d = np.array([
                self.alpha * m_hat /(np.sqrt(v_hat) + self.eps) for m_hat, v_hat in zip(self.m_hat[1::], self.v_hat[1::])
            ])

            x = np.linspace(*x_lim, n_points)
            y = np.linspace(*y_lim, n_points)
            X, Y = np.meshgrid(x, y)
            Z = self.fun([X,Y])

            fig, ax = plt.subplots(figsize=figsize)
            ax.contour(X, Y, Z, n_con, cmap='viridis')
            ax.scatter(*minimum, c = 'black', label = 'min')
            scatter = ax.scatter(None,
                                None,
                                label= label,
                                c=c) 
            
            quiver = ax.quiver(0, 0, 0, 0, width=0.005, linewidth=0.5,
                               angles='xy', scale_units='xy', scale=5,#width=0.005, linewidth=0.5,
                               color='r', label=sign+'$\\alpha \\hat{m_t}/(\\hat{v_t} + \\epsilon)$')
            
            ax.legend(prop={"size": 25})
            
            s = -1 if sign == '-' else 1

            def animate(i):
                current_x = path[i]
                current_d = sc*d[i]
                
                scatter.set_offsets(path[:i, :])
                quiver.set_offsets(np.array([current_x[0], current_x[1]]))
                quiver.set_UVC(s*current_d[0], s*current_d[1])

                title = f"i: {i} / {self.t}/{step} | d{i} = {current_d}/{sc}."
                ax.set_title(title)
            ms_per_frame = 1000 * n_seconds / path_length
            anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)
            plt.show()
        except Exception as e :
            print(e)
            return
        return anim
