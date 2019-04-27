(defproject dl4j-word2vec-in-clojure "0.1.0-SNAPSHOT"
  :description "Deep Learning for Java's Word2Vec walkthrough, in Clojure"
  :url "TODO"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.nd4j/nd4j-native-platform "0.9.1"]
                 [org.deeplearning4j/deeplearning4j-core "0.9.1"]
                 [org.deeplearning4j/deeplearning4j-nn "0.9.1"]
                 [org.deeplearning4j/deeplearning4j-nlp "0.9.1"]]
  :main ^:skip-aot dl4j-word2vec-in-clojure.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}}
  :jvm-opts ["-Xms1024m" "-Xmx10g" "-XX:MaxPermSize=2g"])
