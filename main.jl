using Pkg
Pkg.activate(".")

using Random
using Distributions
using CSV
# warning TextAnalysis has similar functions as TopicsModelsVB
using TextAnalysis
using TopicModelsVB
using Languages
using DataFrames
using Clustering
Random.seed!(7);

path = "/Users/andrecalerovaldez/r_workspace/projects/infoXpand/testGoogleCloudData/data/merged.csv"

isfile(path)

limit_size = 20_000

rawdata = CSV.read(path, DataFrame; limit = limit_size )

subset_data = rawdata[1:limit_size, :]
document_data = Vector{StringDocument}()

for i in 1:nrow(subset_data)
    rawdata[i, :]
    doc = StringDocument(subset_data[i, :text])
    language!(doc, Languages.German())
    author!(doc, string(subset_data[i, :author_id]))
    timestamp!(doc, string(subset_data[i, :created_at]))
    prepare!(doc, strip_punctuation)
    prepare!(doc, strip_articles)
    prepare!(doc, strip_indefinite_articles)
    prepare!(doc, strip_definite_articles)
    prepare!(doc, strip_prepositions)
    prepare!(doc, strip_pronouns)
    prepare!(doc, strip_stopwords)
    prepare!(doc, strip_numbers)
    prepare!(doc, strip_non_letters)
    prepare!(doc, strip_sparse_terms)
    prepare!(doc, strip_frequent_terms)
    prepare!(doc, strip_html_tags)
    push!(document_data, doc)
end

crps = TextAnalysis.Corpus(document_data)



remove_corrupt_utf8!(crps)
remove_case!(crps)
crps
update_inverse_index!(crps)
update_lexicon!(crps)


crps[1]

lex = lexicon(crps)
keys(lex)

sort(collect(lex), by = x -> x.second, rev = true)


m = DocumentTermMatrix(crps)

# clustering tf-idf vectors
D = dtm(m, :dense)
T = tf_idf(D)
cl = kmeans(D, 5)

mx = findmax(T[1,:])

vs = collect(keys(lex))
vs[mx[2]]


a = assignments(cl)
c = counts(cl)

# Topic Modeling
k = 10
iterations = 1000 
α = 0.1 
β  = 0.1

ϕ, θ  = lda(m, k, iterations, α, β)


ϕ
θ




corp = readcorp(:nsf) 

corp.docs = corp[1:5000];
fixcorp!(corp, trim=true)
## It's strongly recommended that you trim your corpus when reducing its size in order to remove excess vocabulary. 

## Notice that the post-fix vocabulary is smaller after removing all but the first 5000 docs.

model = LDA(corp, 9)

train!(model, iter=150, tol=0)
## Setting tol=0 will ensure that all 150 iterations are completed.
## If you don't want to compute the ∆elbo, set checkelbo=Inf.

## training...

showtopics(model, cols=1, 20)


model.beta

println(round.(topicdist(model, 1), digits=3))

showdocs(model, 1)