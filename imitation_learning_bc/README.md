Here, you can run `expert_recorder.py` to record expert demonstrations, expert being you!. 
Right now it uses `MountainCar-v0` environment where you have only 2 actions - right or left, to climb a mountain. 
You can guide by using `a` and `d` for right and left respectively. You can change environment according to your ease too!

To run behaviour cloning in action, first run `python3 expert_recorder.py ./data/`. 
Select the terminal window in which you ran the command, 
then use the `a` and `d` keys to move the agent left and right respectively. 

Once you've finished recording press `+` to save the data to the folder specified. 

Run the model `python3 complete.py ./data/`. It should learn from data and work instantly. 

You can see logs from: `tensorboard --logdir ./logs`


I have borrowed this code from https://github.com/MadcowD/tensorgym. 
