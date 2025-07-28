from dataset_agent import search_kaggle_and_download
from researcher import Agent2


def main():
    input = input("Enter your research question: ")
    search_kaggle_and_download(input, "datasets") # we need to use this dataset and send it to agent 2
    agent = Agent2()
    agent.analyze_hypothesis(input, "datasets/study_scores.csv")
    print(agent.analyze_hypothesis(input, "datasets/study_scores.csv"))
    agent = Agent2()
    agent.analyze_hypothesis(input, "datasets/study_scores.csv")
    print(agent.analyze_hypothesis(input, "datasets/study_scores.csv"))
    
    
    










