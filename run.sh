# run
# python3 main.py --inference inference_method --predictions-file predictions_file --data data_directory

python3 main.py --inference gibbs-sampling --predictions-file my.gibbs_sampling.topics --data datasets/my/


# • --num-topics The number of topics to uncover in the dats. By default, this is set to 10.
# • --alpha The Dirichlet parameter for document-portions (θd). By default this param- eter is set to 0.1.
# • --beta The Dirichlet parameter for topic-word distribution (φk). By default this parameter is set to 0.01.
# • --iterations The number of iterations to use. By default, this value is 100.