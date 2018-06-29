import matplotlib.pyplot as plt
import importlib

agents = ['q_lambda', 'q_lambda_naive', 'q_retrace']

for agent in agents:
    agent_module = importlib.import_module('agent_' + agent)
    test_agent = agent_module.Agent(environment='maze-random-30x30-plus-v0', max_step=2000)
    total_reward, total_step = test_agent.learn()
    plt.plot(total_step)

plt.xlabel('Epoch')
plt.ylabel('Total reward')
plt.legend(['Q(lambda)', 'naive Q(lambda)', 'Retrace(sigma)'], loc='lower right')
plt.title('Result comparison')
plt.show()
