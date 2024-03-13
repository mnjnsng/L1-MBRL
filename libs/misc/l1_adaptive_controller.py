import numpy as np
from scipy.linalg import null_space, inv, expm

class LTISystem(object):
    def __init__(self,A,B,C,D=0.0):
        """Initialize the system."""
        
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        m = self.A.shape[0]
        self.x = np.zeros((m,1))
        # self.x = np.zeros((3,1))
        # self.x = np.zeros((6,1))

    def get_next_state(self,u,dt=0.001):
        state_change = self.A.dot(self.x) + self.B.dot(u)
        self.x += dt*state_change
        output = self.C.dot(self.x) + self.D*u
        return output

class L1_adapt(object):

    def __init__(self, env, f, g, x, wc=350, Ts=0.008):
        '''
        xdot = f(x) + g(x)u (control-affine structure)
        f: mapping from state space to state space (R^n)
        g: mapping from state space to R^(n x m) such that g(x)u makes R^n
        wc: cutoff frequency used in lowpass filter
        Ts: sampling time used in piece-wise continuous adaptation law
        '''
        # plant
        self._env = env
        self.Ts = self._env.wrapped_env.env.dt  # sampling period
        self.f = f/self.Ts
        self.g = g/self.Ts

        self.g_perp = null_space(self.g.T)
        self.time = 0

        # Initialization of state, error and input vectors
        self.x = x[..., np.newaxis]
        self.x_tilde = np.zeros((self._env.observation_space.shape[0], 1))
        self.u = np.zeros((self._env.action_space.shape[0], 1))
        self.n = self.g.shape[0]
        self.m = self.g.shape[1]

        # low pass filter
        self.wc = wc  # cutoff frequency
        self.lpf = LTISystem(A=-self.wc*np.eye(self.m), B=np.eye(self.m), C=self.wc*np.eye(self.m))
        # self.lpf = LTISystem(A=np.array([-self.wc]), B=np.array([1]), C=np.array([self.wc]))
        # self.lpf = LTISystem(A=-self.wc*np.eye(6), B=np.eye(6), C=self.wc*np.eye(6))
        # self.lpf = LTISystem(A=-self.wc*np.eye(2), B=np.eye(2), C=self.wc*np.eye(2))
        # self.lpf = LTISystem(A=-self.wc*np.eye(3), B=np.eye(3), C=self.wc*np.eye(3))

        

        # Initialize parameters needed for L1 controller
        # Choice of Hurwitz matrix used in piece-wise constant adaptation
        self.As = -np.eye(self.n)
        # Initialization of predicted state vector
        self.x_hat = self.x
        self.y_old=0.0

    def low_pass_filter(self,x_new):

        #alpha = dt / (dt + 1 / (2 * np.pi * cutoff))
        alpha=0.2
        y_new = x_new * alpha + (1 - alpha) * self.y_old
        return y_new

    def update_error(self):
        return self.x_hat-self.x

    def adaptive_law(self, x_tilde):

        mat_expm = expm(self.As*self.Ts)
        Phi = inv(self.As) @ (mat_expm - np.eye(self.n))
        adapt_gain = -inv(Phi)@ mat_expm

        gg = np.concatenate(
            (self.g, self.g_perp), axis=1)  # [g,g_perp]

        sigma_hat = inv(gg) @ adapt_gain @ x_tilde
        sigma_hat_m = sigma_hat[:self.m]
        sigma_hat_um = sigma_hat[self.m:]

        return sigma_hat_m, sigma_hat_um

    def state_predictor(self, x, u, sigma_hat_m, sigma_hat_um):

        x_hat_dot = self.f + self.g @(u+sigma_hat_m)+np.matmul(
            self.g_perp, sigma_hat_um)+np.matmul(self.As, self.x_tilde)
        
        x_hat = self.x_hat + x_hat_dot*self.Ts  # Euler integration
        return x_hat

    def get_control_input(self, x, u_bl):

        self.x = x[..., np.newaxis]

        self.x_tilde = self.update_error()
        
        sigma_hat_m, sigma_hat_um = self.adaptive_law(self.x_tilde)
        
        u_l1 = -self.lpf.get_next_state(sigma_hat_m)
        #u_l1 = -sigma_hat_m

        u = u_bl+u_l1.squeeze(-1)

        u=np.clip(u,self._env.wrapped_env.action_space.low,self._env.wrapped_env.action_space.high)

        l1_metadata = {'sigma_hat_m': sigma_hat_m, 'sigma_hat_um': sigma_hat_um, 'xhat': self.x_hat, 'x':self.x }
        u=np.expand_dims(u,axis=-1)

        self.x_hat = self.state_predictor(
            x[..., np.newaxis], u, sigma_hat_m, sigma_hat_um)

        return u.squeeze(-1) , l1_metadata
