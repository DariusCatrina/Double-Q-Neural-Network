from agent import Agent
import numpy as np
import gym


ENVIROMENT_NAME = 'LunarLander-v2'#'CartPole-v1'
MODEL_NAME = ENVIROMENT_NAME + '-policy'

env = gym.make(ENVIROMENT_NAME)

TRAINING_EPISODES = 1500
EVAL_EPISODES = 8



done = False
agent = Agent(env.action_space, env.observation_space, memory_size=50000, name=MODEL_NAME, lr=0.001, update_target_cntr=10)
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


if __name__ == '__main__':
    #training_func()
    eval_func()
