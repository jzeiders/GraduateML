https://liangfgithub.github.io/Proj/F24_Proj2.html



## 9/16
Got the basic pipeline working but forward forecasting is the next big step I need to figure out / understand.
The trick is that I can't use the standard pipelines when the inputs depend on the prior predictions.

I also want to clean up the column selection logic, it's a little redundant between numericals and categoricals. 

Current algorithm absolutely blows. Possibly an analysis issue though, it's actually so bad lol. I do think the actual main thing is to get the features right for rolling and lagging. I wonder if they actually taught anything about this?
Possibly 


Dropping the year because everything is in the same year


Breaking out different models for each store & department does pretty well.

Got absolutely unhinged values with my cheater model with PCA. Adding Lasso Regression solved that quite well. Still need to add a quadratic term on year.

Stuck on the unhinged values, trying to add a break point to see why that is. Maybe I'm fitting on no-data? Need to understand that behavior?

So the SVD issue is about that i'm never skipping anything.

In a much better spot! Got the WMAE down to  1598.460832 by setting negative predictions to zero. Going 
to try scaling to the mean for the dept. Zero was better than mean over 2 or mean for default.