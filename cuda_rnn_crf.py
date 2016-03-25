# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:45:58 2016

@author: edvinj
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import linear #, rnn_cell, rnn
#from tensorflow.models.rnn import seq2seq
#from tensorflow.models.rnn.ptb import reader

flags = tf.flags
logging = tf.logging

#flags.DEFINE_string(
 #   "model", "small",
 #   "A type of model. Possible options are: small, medium, large.")
#flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS

WV_SIZE = 500
NUM_CLASSES = 5

class CUDARNNCRF(object):
    """A RNN cuda model"""


    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        WV_SIZE = config.WV_SIZE
        self.output_size = config.output_size
        self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps*WV_SIZE])
        self._targets = tf.placeholder(tf.float32, [batch_size*num_steps,self.output_size])
        
        
        #cell = BasicRNNCell(size)
        cell = GRUCell(size)
        CRF = CRFCell(config.num_classes,batch_size,session)
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        #self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
        	inputs = tf.split(
        		1, num_steps, self._input_data)
        	#inputs = [tf.squeeze(input_, p[1]) for input_ in inputs]

        #if is_training and config.keep_prob < 1:
        #	inputs = [tf.nn.dropout(input_, config.keep_prob) for input_ in inputs]
        #outputs, state = rnn.rnn(cell, inputs,dtype=tf.float32)
        outputs =[]
        state = self._initial_state
        with tf.variable_scope("CUDA_FWD"):
            for time_step, input_ in enumerate(inputs):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(input_, state)
                outputs.append(cell_output)
        outputs_rev =[]
        state = self._initial_state
        with tf.variable_scope("CUDA_BKD"):
            for time_step, input_ in enumerate(reversed(inputs)):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(input_, state)
                outputs_rev.append(cell_output)
        
        outputs = [a+b for a, b in zip(outputs,outputs_rev)]
        self.output = output = tf.reshape(tf.concat(1, outputs), [-1, size])        
        self.logits = logits = tf.nn.xw_plus_b(output,
                                 tf.get_variable("W_out",[size,self.output_size]),
                                 tf.get_variable("b_out", [self.output_size]))
        
        
        CRF._obs = self.logits
        logits_crf = CRF()
        logits_crf = [item for logits_crf_ in logits_crf for item in logits_crf_]
        self.logits_crf = tf.reshape(tf.pack(logits_crf), [-1,self.output_size])
        loss = 0.
        
        loss = tf.nn.softmax_cross_entropy_with_logits(self.logits_crf,self._targets)
        print("here")
        self._final_state = state
        self._cost = cost = tf.reduce_sum(loss)
        print("cost",cost)
        if not is_training:
            self.softmax = tf.nn.softmax(self.logits_crf)
            return
            
        self.lr = tf.Variable(0.01, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        
        def assign_lr(self, session, lr_value):
            session.run(tf.assign(self.lr, lr_value))

        @property
        def input_data(self):
            return self._input_data
        
        @property
        def targets(self):
            return self._targets
        
        @property
        def initial_state(self):
            return self._initial_state
        
        @property
        def cost(self):
            return self._cost
        
        @property
        def final_state(self):
            return self._final_state
        
        @property
        def lr(self):
            return self._lr
        
        @property
        def train_op(self):
            return self._train_op
        
        



class RNNCell(object):
    
    """RNNCell"""
    
    def __call__(self,inputs,state):
        
        raise NotImplementedError("Abstract method")
        #return tf.tanh(linear.linear(input,hidden))        
    
    @property
    def input_size(self):
        """ size of input to this cell"""
        raise NotImplementedError("Abstract method")   
    
    @property
    def state_size(self):
        """size of hidden state"""
        raise NotImplementedError("Abstract method")
        
    @property
    def output_size(self):
        """ size of output of cell"""
        raise NotImplementedError("Abstract method")
            
    def zero_state(self, batch_size, dtype):
        """ create a hidden layer of zeros"""
        zeros = tf.zeros(tf.pack([batch_size,self.state_size]),dtype=dtype)
        return tf.reshape(zeros,[-1,self.state_size])
        
        
class BasicRNNCell(RNNCell):
    
    def __init__(self,num_units):
        
        self._num_units = num_units
    
    @property    
    def input_size(self):
        return self._num_units
    
    @property    
    def output_size(self):
        return self._num_units
    
    @property    
    def state_size(self):
        return self._num_units
        
    
    def __call__(self,inputs,state):
        output = tf.tanh(linear.linear([inputs,state],self.output_size,False))
        return output, output
        

class GRUCell(RNNCell):
    
    """Gated Recurrent Cell"""
    
    def __init__(self,num_units):
        
        self._num_units = num_units
        
    @property    
    def input_size(self):
        return self._num_units
    
    @property    
    def output_size(self):
        return self._num_units
    
    @property    
    def state_size(self):
        return self._num_units
        
        
    def __call__(self,inputs, state, scope=None):
        
        with tf.variable_scope(scope or type(self).__name__):
            
            with tf.variable_scope("Gates"):
                r, u = tf.split(1,2,linear.linear([inputs,state],
                                                  self._num_units * 2,
                                                  True, 1.0))
                                                  
                r, u = tf.sigmoid(r), tf.sigmoid(u)
            
            with tf.variable_scope("Candidate"):
                c = tf.tanh(linear.linear([inputs,state * r],self._num_units,
                                          True))
            new_h = u * state + (1-u) * c
            
            return new_h, new_h
            
            
class CRFCell(object):
    
    def __init__(self,num_classes,num_examples,session):
        self._num_classes = num_classes
        self._num_examples = num_examples
        #self._obs = #tf.placeholder(tf.float32,shape=[num_examples,num_classes])
        self._session = session
    def __call__(self,scope=None):
        observations = tf.split(0,self._num_examples,self._obs)
        observations = [tf.squeeze(obs_) for obs_ in observations]
        #print(len(observations))
        #print(observations[0].get_shape())
        smoothed = []
        with tf.variable_scope(scope or type(self).__name__) as scope:
            for n, obs in enumerate(observations):
                if n > 0:
                    scope.reuse_variables()
                fwd = forward(obs,self._num_classes,session)
                bkwd = backward(obs,self._num_classes,session)
                smoothed.append(smooth(fwd,bkwd,session))
                
        return smoothed
        
        
def forward(obs,m,scope=None):
    init = np.array([1./m]*m*m).astype('float32').reshape(m,m)
    output = []
    output.append(init)
    obs = tf.split(0,m,obs)
    obs = [tf.squeeze(obs_) for obs_ in obs]
    with tf.variable_scope(scope or "Backward") as scope:
        T = tf.get_variable("Transitions",[m,m],
                            initializer=tf.random_uniform_initializer(-1,1))
        #session.run(tf.initialize_all_variables())
        for time_step, o_ in enumerate(obs):
            #print(time_step)
            
            if time_step > 0:
                #scope.reuse_variables()
                init = output[time_step]
            O = tf.diag(o_)
            
            tmp = tf.matmul(O,T)
            
            tmp = tf.matmul(tmp,init)
            
            tmp = tf.div(tmp,tf.reduce_sum(tmp))*2
            #print(tmp.get_shape())
            output.append(tmp)
    return output


def backward(obs,m,scope=None):
    init = np.array([1.]*m*m).astype('float32').reshape(m,m)
    output = []
    output.append(init)
    obs = tf.split(0,m,obs)
    obs = [tf.squeeze(obs_) for obs_ in obs]
    with tf.variable_scope(scope or "Forward") as scope:
        T = tf.get_variable("Transitions",[m,m],
                            initializer=tf.random_uniform_initializer(-1,1))
        #session.run(tf.initialize_all_variables())
        for time_step, o_ in enumerate(reversed(obs)):
            #print(time_step)
            
            if time_step > 0:
                #scope.reuse_variables()
                init = output[time_step]
            O = tf.diag(o_)
            
            tmp = tf.matmul(O,T)
            
            tmp = tf.matmul(tmp,init)
            
            tmp = tf.div(tmp,tf.reduce_sum(tmp))*2
            #print(tmp.get_shape())
            output.append(tmp)
    return output


def smooth(fwd,bkwd,scope=None):
    #init_fwd = np.array([1./m]*m*2).astype('float32').reshape(m,m)
    #init_bwd = np.array([1.]*m*2).astype('float32').reshape(m,m)
    output = []
    for time_step,(f_,b_) in enumerate(zip(fwd,reversed(bkwd))):
        tmp = tf.mul(f_,b_)
        tmp = tf.div(tmp,tf.reduce_sum(tmp))*2
        output.append(tmp)
    return [output_[:,0] for output_ in output[1:]]

    
def linear2(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  assert args
  if not isinstance(args, (list, tuple)):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope or "Linear") as scope:
    try:
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        
    except:
        scope.reuse_variables()
        matrix = tf.get_variable("Matrix")
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable("Bias", [output_size],
                                initializer=tf.constant_initializer(bias_start))
  return res + bias_term
        





class Config(object):
    """ Configuration to be past to CUDARNN"""
    
    def __init__(self, batch_size,
                 num_steps,
                 hidden_size,
                 WV_SIZE,
                 output_size,
                 num_classes,
                 max_grad_norm):
                     
                     
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.WV_SIZE = WV_SIZE
        self.output_size = output_size
        self.max_grad_norm = max_grad_norm
        self.num_classes = num_classes
        
    @classmethod
    def create_default(cls):
        return cls(50,5,50,5,5,5,5)
        


