from agent import Agent
import numpy as np
import gym
import torch

env_names = ['CartPole-v1', 'LunarLander-v2','MountainCar-v0','Acrobot-v1' ]
ENVIROMENT_NAME = env_names[1]
MODEL_NAME = ENVIROMENT_NAME + '-policy'

env = gym.make(ENVIROMENT_NAME)

TRAINING_EPISODES = 3000
EVAL_EPISODES = 8

memory_size = 10000

done = False
agent = Agent(env.action_space, env.observation_space, memory_size=memory_size, name=MODEL_NAME, lr=0.001, update_target_cntr=25)
scores = []
avg_score = None



#training....
def training_func():
    global done, agent, scores, avg_score, TRAINING_EPISODES


    for episode in range(TRAINING_EPISODES):
        state = env.reset()
        score = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward

            agent.store_transition(state=state, next_state=next_state,action=action,reward=reward,done=done)
            agent.learn()

            state = next_state

        done=False
        scores.append(score)

        agent.update(episode, score)
        avg_score = np.mean(scores[-5:])

        if episode % 25 == 0:
            print('Episode {} and average score:{}(eps: {}), with the best score of:{}'.format(episode, avg_score, agent.eps, agent.best_score))



#evaluation
def eval_func():
    global MODEL_NAME, EVAL_EPISODES
    score = 0
    done = False
    agent.eval(MODEL_NAME)

    for episode in range(EVAL_EPISODES):
        state = env.reset()
        score = 0

        while not done:
            env.render()

            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward

        print("Finnished episode in: {}".format(score))
        score = 0
        done = False

def random_agent():
    for i_episode in range(3):
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)


if __name__ == '__main__':
    #training_func()
    #random_agent()
    eval_func()
