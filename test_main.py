from agent_q_lambda_naive import Agent
import matplotlib.pyplot as plt

test_agent = Agent()
total_reward,total_step = test_agent.learn()

plt.plot(total_step)
plt.xlabel('Epoch')
plt.ylabel('Total reward')
plt.show()
