import matplotlib.pyplot as plt
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel

# Compute coherence scores across different number of topics
def compute_coherence(dictionary, corpus, texts, limit, start=2, step=5):
    coherence_val = []
    for num_topics in range(start, limit, step):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                       id2word=dictionary,
                                       num_topics=num_topics,
                                       alpha='auto',
                                       passes=10)

        coherencemodel_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_val.append(coherencemodel_lda.get_coherence())

    return coherence_val

def plot_coherence_values(coherence_values, start, limit, step, plot_title):
    x = range(start, limit, step)
    y_lda = coherence_values
    x = list(range(start, limit, step))

    # Find the index of the highest coherence score
    max_index = y_lda.index(max(y_lda))  
    max_x = x[max_index]  # Get the corresponding number of topics
    max_y = max(y_lda)  # Get the highest coherence score
    print(f"Highest coherence score: {max_y:.4f} at {max_x} topics")

    # Plot the coherence score curve
    plt.plot(x, y_lda, label={plot_title}, marker='o')

    # Annotate the highest point
    plt.annotate(f'Max: {max_y:.4f}, at {max_x} topics', 
                xy=(max_x, max_y), 
                xytext=(max_x + 2, max_y - 0.02),  
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=10, color='red')

    plt.xlabel("Num Topics")
    plt.ylabel("Coherence Score")    
    plt.legend()

    plt.savefig('{plot_title}.png')
    plt.show()