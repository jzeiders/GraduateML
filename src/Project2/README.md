https://liangfgithub.github.io/Proj/F24_Proj2.html



## 9/16
Got the basic pipeline working but forward forecasting is the next big step I need to figure out / understand.
The trick is that I can't use the standard pipelines when the inputs depend on the prior predictions.

I also want to clean up the column selection logic, it's a little redundant between numericals and categoricals. 

Current algorithm absolutely blows. Possibly an analysis issue though, it's actually so bad lol. I do think the actual main thing is to get the features right for rolling and lagging. I wonder if they actually taught anything about this?
Possibly 
