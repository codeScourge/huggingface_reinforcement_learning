from stable_baselines3 import A2C
import gymnasium as gym


# --- env
weights_path = "./weights/04"

resp = input("Would you like to train (1) or evaluate (2) > ")
training:bool = False if resp == "2" else True
render_mode = None if training else "human"
env = gym.make("MountainCar-v0", goal_velocity=0.1, render_mode=render_mode)


# --- training
if training:
    # by wrapping the environment it can "read" the observation / action format and adapt 
    # the input/output layers. This is one type of architecture
    # - "Multi-Layer-Perceptron" policy - used for vector-based observations
    # - CNNPolicy should be used for image inputs 
    # - MultiInputPolicies
    policy_name = "MlpPolicy"


    # we could also build our own torch.nn.Module and "import" it into our A2C wrapper by using IT instead of the string
    # given we implement a forward pass that follows the format of A2C
    # input: observations x
    # output: action_logits, predicted_value
    model = A2C(
        policy_name,
        env,
        verbose=1
    )

    # since it is just a wrapper around a pytorch model, we can extract the weights using `.state_dict()`
    print(model.policy)
    
    model.learn(total_timesteps=30000)
    model.save(weights_path)
    
else:
    model = A2C.load(
        weights_path,
        env
    )
    
    # vectorizing an environment, means putting multiple separate copies in a vector
    # that way we can get observations in batches and take actions in parallel to train faster
    # besides that they also handle resets automatically (to make sure observations are always fresh, 
    # preventing delays for our parallel processing)
    vec_env = model.get_env()
    obs = vec_env.reset()

    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render(render_mode)