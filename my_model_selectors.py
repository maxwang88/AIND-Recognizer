import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    p is the number of estimated parameters.
    How to compute 'p'? https://stats.stackexchange.com/questions/12341/number-of-parameters-in-markov-model
    N is the number of samples.
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_num_components = None
        lowest_BIC = None
        for n_component in range(self.min_n_components, self.max_n_components+1):
            try:
                N = len(self.X)
                # p = transition probabilities + emission probabilities
                # transition probs = (number of states) * (number of states - 1)
                # emission probs = (number of states) * (number of features) * 2, since this is gaussian hmm, 
                # two variables for each feature in each state represent mean and variance.

                #######Correction#######
                # "Free parameters" are parameters that are learned by the model and it is a sum of:
                # 1. The free transition probability parameters, which is the size of the transmat matrix less one row because they add up to 1 and therefore the final row is deterministic, so `n*(n-1)`
                # 2. The free starting probabilities, which is the size of startprob minus 1 because it adds to 1.0 and last one can be calculated so `n-1`
                # 3. Number of means, which is `n*f`
                # 4. Number of covariances which is the size of the covars matrix, which for "diag" is `n*f`
                p = len(self.X[0]) * n_component * 2 + n_component * (n_component - 1) + (n_component - 1)
                logL = self.base_model(n_component).score(self.X, self.lengths)
                BIC = -2 * logL + p * np.log(N)
                if lowest_BIC is None or lowest_BIC > BIC:
                    lowest_BIC = BIC
                    best_num_components = n_component
            except:
                None

        if best_num_components is None:
            return self.base_model(self.n_constant)

        return self.base_model(best_num_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    in which M represents number of words,
    log(P(X(i)) represents the log-likelihood of the fitted model for the current word,
    1/(M-1)SUM(log(P(X(all but i)) represents the average of the log-likelihoods of the fitted models for all the other words
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_num_components = None
        highest_DIC = None

        words = list(self.words.keys())
        words.remove(self.this_word)
        M = len(words)
        sum_log_P_X_not_i = 0

        for n_component in range(self.min_n_components, self.max_n_components):
            try:
                log_P_Xi = self.base_model(n_component).score(self.X, self.lengths)

                for word in words:
                    try:
                        model_selectors = ModelSelector(self.words, self.hwords, word, self.n_constant, self.min_n_components, 
                        self.max_n_components, self.random_state, self.verbose)
                        sum_log_P_X_not_i += model_selectors.base_model(n_component).score(model_selectors.X, model_selectors.lengths)
                    except:
                        M -= 1

                DIC = log_P_Xi - sum_log_P_X_not_i / (M - 1)

                if highest_DIC is None or highest_DIC < DIC:
                    highest_DIC = DIC
                    best_num_components = n_component
            except:
                None

        if best_num_components is None:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(best_num_components)





class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    N_FOLD = 3

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score, best_model = None, None
        
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            scores, n_folds = [], SelectorCV.N_FOLD
            model, logL = None, None
            
            if(len(self.sequences) < n_folds):
                break
            
            split_method = KFold(random_state=self.random_state, n_splits=n_folds)
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                X_train, train_length = combine_sequences(cv_train_idx, self.sequences)
                X_test, test_length  = combine_sequences(cv_test_idx, self.sequences)
                try:
                    model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                    random_state=inst.random_state, verbose=False).fit(X_train, train_length)
                    logL = model.score(X_test, test_length)
                    scores.append(logL)
                except:
                    break
            
            
            avg = np.average(scores) if len(scores) > 0 else float("-inf")
            
            if best_score is None or avg > best_score:
                best_score, best_model = avg, model
        
        if best_model is None:
            return self.base_model(self.n_constant)
        else:
            return best_model
        