# The module plots high dimensional vectors in 2 or 3 dimensional plots.


       Input args:
        
            - input_dict must be like: {'comedy' : [numpy_matrix, label_number, numpy_array_of_file_names],
                                        'action' : [numpy_matrix, label_number, numpy_array_of_file_names],
                                         .................................................................,
                                        }
            - by default dict keys are used in plot as class labels. keys might be helpful, but you can exclude them in plot
              by setting 'dict_keys_as_labels' to False. you could use any input_dict key (just don't dublicate them), 
              it's for your attention to see what classes you added and their labels;
            - 'numpy_matrix' must be 2 dimentional, samples in rows and their features in cols;
            - 'label_number' is just label integer. That integers are used at plot to generate colors. The numbers must be started
              from zero;
            - 'numpy_array_of_file_names' must be a numpy vector of strings like : np.array(["comedy", "action", ...])
