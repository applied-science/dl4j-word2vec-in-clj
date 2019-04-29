(ns dl4j-word2vec-in-clojure.workbook
  "Port of DL4J's Word2Vec tutorial from Java to Clojure
   See the original at https://deeplearning4j.org/docs/latest/deeplearning4j-nlp-word2vec
   Follow along with my write-up at TODO"
  (:require [clojure.string :as string]
            [clojure.java.io :as io])
  (:import [org.deeplearning4j.text.sentenceiterator BasicLineIterator]
           [org.deeplearning4j.models.word2vec Word2Vec$Builder]
           [org.deeplearning4j.text.tokenization.tokenizer.preprocessor CommonPreprocessor]
           [org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory]
           [org.deeplearning4j.models.embeddings.loader WordVectorSerializer]
           [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.api.buffer DataBuffer$Type]
           [org.datavec.api.util ClassPathResource]
           [org.deeplearning4j.plot BarnesHutTsne$Builder]))


;;
;;
;; Please note that this workbook is intended as a supplement to
;; "Exploring Word2Vec in Clojure with DL4J" at TODO. The code here
;; will not make much sense on its own, because it is assumed you are
;; evaluating sexps here merely to understand the contents of that
;; blog post.
;;
;;


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;; Building the Model
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Peek at the toy dataset
(let [lines (string/split-lines (slurp "resources/raw_sentences.txt"))]
  (->> lines
       (drop (rand-int (count lines)))
       (take 10)))

;; Build word2vec model
(def model ;; `vec` in the original Java
  (-> (Word2Vec$Builder.)
      (.minWordFrequency 5)
      (.iterations 1)
      (.layerSize 100)
      (.seed 42)
      (.windowSize 5)
      (.iterate (BasicLineIterator. "resources/raw_sentences.txt"))
      (.tokenizerFactory (doto (DefaultTokenizerFactory.)
                           (.setTokenPreProcessor (CommonPreprocessor.))))
      (.build)))

;; Build word2vec model over toy dataset
;; NB: may take a few seconds
(.fit model)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;; Evaluating the Model
;;;;
;;;; NB: all model-based results are nondeterministic, so the example
;;;; evaluation values are only approximate. Only if your results are
;;;; WAY off should you suspect an error.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(.getWordVectorMatrix model "day")
;; #object[org.nd4j.linalg.cpu.nativecpu.NDArray 0x59073fb5 "[0.41,  0.21,  0.15,  -0.21,  -0.04,  -0.40,  -0.12,  -0.10,  -0.32,  0.35,  0.21,  0.28,  0.12,  -0.07,  0.05,  -0.07,  -0.20,  0.21,  0.14,  -0.15,  0.07,  0.20,  0.42,  -0.23,  0.10,  -0.40,  0.11,  -0.42,  -0.19,  -0.11,  0.29,  -0.00,  0.46,  -0.51,  0.14,  -0.23,  0.08,  -0.21,  -0.07,  0.10,  -0.31,  -0.19,  0.11,  0.21,  -0.07,  -0.12,  -0.47,  -0.16,  0.16,  -0.14,  0.28,  0.04,  0.24,  -0.14,  -0.35,  0.09,  -0.24,  -0.07,  0.16,  -0.46,  -0.28,  -0.01,  0.15,  0.43,  0.16,  0.04,  0.04,  0.19,  -0.25,  -0.35,  0.24,  -0.06,  0.18,  -0.01,  -0.03,  0.10,  0.06,  0.11,  0.13,  0.04,  -0.03,  -0.19,  -0.45,  0.12,  -0.00,  0.04,  0.17,  -0.34,  -0.03,  -0.18,  -0.11,  0.01,  0.15,  -0.06,  -0.19,  0.25,  0.01,  0.28,  -0.32,  -0.11]"]

;; Closest words to 'day':
(.wordsNearest model "day" 10)
;; ["night" "week" "year" "game" "season" "group" "time" "office" "-" "director"]

;; Cosine similarity of 'day' and 'night':
(.similarity model "day" "night")
;; 0.7328975796699524

(.wordsNearest model "man" 10)
;; ["program" "company" "director" "market" "political" "group" "general" "such" "family" "business"]

;; Toy dataset doesn't include the words "king" or "queen" so our model can't support the classic "king - queen = man - woman" word vector arithmetic :(
(.wordsNearest model "king" 10)
;; []

(.wordsNearest model "queen" 10)
;; []


;;;; Save our model for later
;; Deprecated old approach:
;; (WordVectorSerializer/writeWordVectors model "serialized-model.txt")

;; Fresh new approach:
(WordVectorSerializer/writeWord2VecModel model "serialized-word2vec-model.zip")



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;; Visualizing saved word vectors
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; We use Barnes-Hut algorithm TSNE to visualize a sample dataset provided by DL4J.

;; Prepare for an n-dimensional array of doubles
(Nd4j/setDataType DataBuffer$Type/DOUBLE)

;; Get the data of all unique word vectors (from the provided dataset)
(def vectors
  (WordVectorSerializer/loadTxt (.getFile (ClassPathResource. "words.txt"))))

;; Separate the weights of unique words into their own list
(def weights
  (.getSyn0 (.getFirst vectors)))

;; Separate strings of words into _their_ own list
(def words "aka `cache` or `cacheList`"
  (map #(.getWord %) (.vocabWords (.getSecond vectors))))

;; I recommend taking a peek at what `words` evaluates to. Be careful
;; doing the same with `weights` or `vectors`; they are big enough to
;; cause IDE slowdowns if you print them to a REPL.

;; Build dual-tree TSNE model
(def words-tsne
  (-> (BarnesHutTsne$Builder.)
      (.setMaxIter 100)
      (.theta 0.5)
      (.normalize false)
      (.learningRate 500)
      (.useAdaGrad false)
      (.build)))

;; Establish the TSNE values
;; NB: careful, it could take a minute.
(.fit words-tsne weights)

;; ...and save them to a file
(io/make-parents "target/tsne-standard-coords.csv")

(.saveAsFile words-tsne words "target/tsne-standard-coords.csv")

;; now use gnuplot to visualize that file


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; now 3d
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; helper fn
(defn build-tsne
  ([] (build-tsne 2))
  ([dims]
   (-> (BarnesHutTsne$Builder.)
       (.setMaxIter 100)
       (.theta 0.5)
       (.normalize false)
       (.learningRate 500)
       (.useAdaGrad false)
       (.numDimension dims)
       (.build))))

(def words-tsne-3d (build-tsne 3))

(.fit words-tsne-3d weights)

(.saveAsFile words-tsne-3d words "target/tsne-standard-coords-3d.csv")

;; now use gnuplot to visualize that file


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;; Visualizing _our_ model
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; We've visualized words.txt, so now we visualize our saved model
;; from earlier.

;; Get the data of all unique word vectors
;; the `true` parameter means "try to read the extended model". For more see https://deeplearning4j.org/api/latest//org/deeplearning4j/models/embeddings/loader/WordVectorSerializer.html#readWord2VecModel-java.io.File-boolean-
(def w2v
  (WordVectorSerializer/readWord2VecModel "serialized-word2vec-model.zip" true))

(def w2v-weights
  (.getSyn0 (.lookupTable w2v)))

(def w2v-words
  (map str (.words (.vocab w2v))))

(def w2v-tsne (build-tsne))

(.fit w2v-tsne w2v-weights)

(io/make-parents "target/tsne-w2v.csv")

(.saveAsFile w2v-tsne w2v-words "target/tsne-w2v.csv")


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; now 3d
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def w2v-tsne-3d (build-tsne 3))

(.fit w2v-tsne-3d w2v-weights)

(.saveAsFile w2v-tsne-3d w2v-words "target/tsne-w2v-3d.csv")




;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;; Importing more comprehensive Models
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; No model built over the toy raw sentences corpus will be useful,
;; because it contains so few words:
w2v-words

;; So we import a more robust model to see Word2Vec's true power.

;; (It may be necessary to reset your data type, depending on your REPL state.)
(Nd4j/setDataType DataBuffer$Type/DOUBLE)

;; Warning: can be compute-heavy (read: sssslllooooowwwww)
(def gnews-vec
  (WordVectorSerializer/readWord2VecModel "/path/to/GoogleNews-vectors-negative300.bin.gz"))

;; Now for some of that famous "word math" using the
;; Google-News-corpus-trained vectors.

;; king:queen::man:[woman, Attempted abduction, teenager, girl] 
(.wordsNearest gnews-vec
               ["queen" "man"] ; "positive" words
               ["king"]        ; "negative" words
               5)

;; China:Taiwan::Russia:[Ukraine, Moscow, Moldova, Armenia]
(.wordsNearest gnews-vec
               ["Taiwan" "Russia"]
               ["China"] 5)

;; Some data clutter:
(.wordsNearest gnews-vec "United_States" 5)



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;; GloVe: Global Vectors
;;;; https://deeplearning4j.org/docs/latest/deeplearning4j-nlp-word2vec#glove
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def glove-vectors
  (WordVectorSerializer/loadTxtVectors (io/file "/path/to/glove.6B.50d.txt")))

(.wordsNearest glove-vectors
               ["queen" "man"] ; "positive" words
               ["king"]        ; "negative" words
               5)



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;; Deeper explorations
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn analogy
  "According to the given `model`, `a` is to `b` as `c` is to the `n` value(s) returned.
  That is, a : b :: c : [return values]. Defaults to n=5 results.
  For example: king:queen::man:[woman, Attempted abduction, teenager, girl]
  (NB: Results may vary across training results, even on the same source corpus.)"
  ([model a b c] (analogy model 5 a b c))
  ([model n a b c]
   (.wordsNearest model
                  [b c] [a] n)))

;; king:queen::man:[woman, Attempted abduction, teenager, girl] 
(analogy gnews-vec "king" "queen" "man")

;; We're going to be running a lot of analogy checks based on the Google News vectors, so let's make a helper function.
(def gnews-analogy (partial analogy gnews-vec))

;; Take it for a spin with variations on the king/queen/man/woman theme.
(gnews-analogy 1 "man" "king" "woman")

(gnews-analogy "king" "prince" "queen")

;; I wonder what GloVe says?
(analogy glove-vectors "king" "queen" "man")

;; Let's try some more example analogies.
(gnews-analogy "Italy" "Rome" "China")

;; Remember to downcase place names with GloVe.
(analogy glove-vectors "germany" "berlin" "china")

;; China:Taiwan::Russia:[Ukraine, Moscow, Moldova, Armenia]
(gnews-analogy "China" "Taiwan" "Russia")

;; house:roof::castle:[dome, bell_tower, spire, crenellations, turrets]
(gnews-analogy "house" "roof" "castle")

;; GloVe picks words that are just as accurate but from a different perspective:
(analogy glove-vectors "house" "roof" "castle")

;; knee:leg::elbow:[forearm, arm, ulna_bone]
(gnews-analogy "knee" "leg" "elbow")


;; The next analogy took some finesse.

;; New York Times:Sulzberger::Fox:[Murdoch, Chernin, Bancroft, Ailes]

;; It requires some exploration to phrase "The New York Times" so the
;; model can recognize it. Try the following expression with a few
;; different variations.
(.wordsNearest gnews-vec "NYT" 10)

;; ...my results don't perfectly match the docs, but it's close enough for casual NLP work:
(gnews-analogy "NYTimes" "Sulzberger" "Fox")

;; love:indifference::fear:[apathy, callousness, timidity, helplessness, inaction]
(gnews-analogy "love" "indifference" "fear")

;; GloVe gives similarly poetic results:
(analogy glove-vectors "love" "indifference" "fear")

;; Donald Trump:Republican::Barack Obama:[Democratic, GOP, Democrats, McCain]
;; "It's interesting to note that, just as Obama and McCain were rivals, so too, Word2vec thinks Trump has a rivalry with the idea Republican."
(gnews-analogy "Donald_Trump" "Republican" "Barack_Obama")

;; monkey:human::dinosaur:[fossil, fossilized, Ice_Age_mammals, fossilization]
(gnews-analogy "monkey" "human" "dinosaur")

;; building:architect::software:[programmer, SecurityCenter, WinPcap]
(gnews-analogy "building" "architect" "software")


;;;; Aside: Replication Crisis
;; I was unable to replicate quite a few of the examples in the DL4J docs. For example:

;; Library - Books = Hall
(.wordsNearest gnews-vec ["Library"] ["books"] 10)
;; ["Wastewater_Treatment_Facility" "Municipal_Building" "Museum"]

;; I trimmed more than 5 misspellings, synonyms for "library", and
;; nonsensical non-words from these results. What I'm left with isn't
;; what the DL4J folks got, but...it's something. I suppose a library
;; without books is indeed like a wastewater treatment facility.

;; Another analogy that failed to replicate:
;; Stock Market ≈ Thermometer
(.wordsNearest gnews-vec "Stock_Market" 10)
;; ["ETF_VTI" "Stockmarket" "Outlook_FairWealth_Securities" "VIX_ETN" "GIB_NYSE" "Suckers_Read" "Bearish_Bets" "Options_Trader" "Index_VTSMX" "Index_EEM_PriceWatch"]

;; I distinctly do not see Thermometer. Maybe in the opposite direction?

(.wordsNearest gnews-vec "Thermometer" 10)
;; ["thermometer" "Thermometers" "Soother" "Digital_Thermometer" "hygrometer" "infrared_thermometer" "Temperature_Transmitter" "Infrared_Thermometer" "Humidifier" "Coffee_Maker"]

;; Are they anywhere near each other?
(.similarity gnews-vec "Stock_Market" "Thermometer")
;; 0.15860738105272534

;; lolno



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;; Domain-specific similarity
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(.similarity gnews-vec "Sweden" "Sweden")

(.similarity gnews-vec "Sweden" "Norway")

(->> (.words (.vocab gnews-vec))
     (take 10000) ;; <-- comment this out for true but compute-heavy
                  ;; results. (For me the whole batch of 3 million
                  ;; words took ~400 seconds.)
     (map (juxt identity #(.similarity gnews-vec "Sweden" %)))
     (sort-by second >)
     (take 10))
;; (["Sweden" 1.0]
;;  ["Finland" 0.8084677457809448]
;;  ["Norway" 0.7706173658370972]
;;  ["Denmark" 0.7673707604408264]
;;  ["Swedish" 0.7404001951217651]
;;  ["Swedes" 0.7133287191390991]
;;  ["Scandinavian" 0.6518087983131409]
;;  ["Stena_Match_Cup" 0.6437666416168213]
;;  ["Netherlands" 0.6401048302650452]
;;  ["official_Lars_Emilsson" 0.6374118328094482])

;; Let's try again, comparing only countries

(def countries-marc
  (set (map (fn [s] (subs s (inc (.indexOf s " "))))
            (string/split-lines (slurp "/Users/daveliepmann/src/data/marc-country-codes.txt")))))

(->> (.words (.vocab gnews-vec))
     (filter countries-marc) ;; ignore non-country words
     (map (juxt identity #(.similarity gnews-vec "Sweden" %)))
     (sort-by second >)
     rest ;; ignore Sweden itself
     (take 10))


;; fin
