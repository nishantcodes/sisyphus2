# Judge gains fold by fold, not against the fold spread

Every experiment is scored on the same frozen folds, so most of the fold-to-fold
spread is *fold difficulty* shared by all models. Comparing a mean gain against
that raw spread throws away real improvements: on the bank proxy a +0.0016 AUC
gain against a 0.0027 spread looked like noise, yet every one of five folds
improved (paired t = 6.3). The ledger's `vs parent` column is the paired mean
delta and t; trust it over eyeballing CV +/- std. A gain whose t is small is
noise even when the mean looks large, and vice versa.
