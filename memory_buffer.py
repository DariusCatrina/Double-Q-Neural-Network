import numpy as np

class MemoryBuffer(object):
    def __init__(self, observation_space, mem_size=50000):
      self.mem_size = mem_size
      self.observation_space = observation_space.shape[0]

      self.curr_state_mem = np.zeros((self.mem_size,self.observation_space),dtype=np.float32)
      self.next_state_mem = np.zeros((self.mem_size,self.observation_space),dtype=np.float32)
      self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
      self.flag_mem = np.zeros(self.mem_size, dtype=np.int8)
      self.action_mem = np.zeros(self.mem_size, dtype=np.int32)

      self.mem_cntr = 0

    def insert(self, state, next_state, reward, action, flag):
      index = self.mem_cntr % self.mem_size
      self.mem_cntr += 1

      self.curr_state_mem[index] = state
      self.next_state_mem[index] = next_state
      self.action_mem[index] = action
      self.reward_mem[index] = reward
      self.flag_mem[index] = int(flag)


    def sample(self, batch_size):
      if batch_size > self.mem_cntr:
        return None

      index = np.random.choice(min(self.mem_cntr, self.mem_size), size=batch_size, replace=True)

      return self.curr_state_mem[index],self.next_state_mem[index],self.action_mem[index],self.reward_mem[index],self.flag_mem[index]

    def show_occupied_memory(self):
        index = np.arange(min(self.mem_cntr, self.mem_size))
        print('States:')
        print(self.curr_state_mem[index] )
        print('Next States:')
        print(self.next_state_mem[index] )
        print('Actions:')
        print(self.action_mem[index] )
        print('Dones:')
        print(self.flag_mem[index] )