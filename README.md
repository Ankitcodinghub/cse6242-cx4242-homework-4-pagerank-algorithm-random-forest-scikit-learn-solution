# cse6242-cx4242-homework-4-pagerank-algorithm-random-forest-scikit-learn-solution
**TO GET THIS SOLUTION VISIT:** [CSE6242/CX4242: Homework 4 : PageRank Algorithm, Random Forest, SciKit Learn Solution](https://www.ankitcodinghub.com/product/cse6242-cx4242-homework-4-pagerank-algorithm-random-forest-scikit-learn-solution/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;89691&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;4&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (4 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;&nbsp;CSE6242\/CX4242: Homework 4 : PageRank Algorithm, Random Forest, SciKit Learn Solution&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (4 votes)    </div>
    </div>
<p class="c15"><span class="c0">In this question, you will implement the PageRank algorithm in Python for a large dataset.</span>

<p class="c15"><span class="c0">The PageRank algorithm was first proposed to rank web search results, so that more “important” web pages are ranked higher. &nbsp;It works by considering the number and “importance” of links pointing to a page, to estimate how important that page is. PageRank outputs a probability distribution over all web pages, representing the likelihood that a person randomly surfing the web (randomly clicking on links) would arrive at those pages.</span>

<p class="c15"><span class="c0">As mentioned in the lectures, the PageRank values are the entries in the dominant eigenvector of the modified adjacency matrix in which each column’s values adds up to 1 (i.e., “column normalized”), and this eigenvector can be calculated by the power iteration method, which iterates through the graph’s edges multiple times to update the nodes’ probabilities (‘scores’ in pagerank.py) in each iteration :</span>

<p class="c15">For each iteration, the Page rank computation for each node would be :

<p class="c15"><img decoding="async" title="" data-src="https://lh6.googleusercontent.com/50WzIj1K6X5_iTK5BNN3aIVQTeOoWTSy_lkf4i-D3z8qIzXyhm4AH6hv7DhcNY5rmGqyxku6w2dqpKn-faIL5ZPLGRLAyMdFadqKTe1Nl4XPvLrYJ6FFAI4cn3RjuW5zVB92X2-w" alt="" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" class="lazyload">

<p class="c15"><span class="c0">Where:</span>

<p class="c15"><img decoding="async" title="" data-src="https://lh3.googleusercontent.com/dDZRuMoJnw6esZISHCXssOsZFpM-nZLowB0xCDB8muoZ3hwrePTmlwKn0J3ctP4uyTT1VOEZs2K32tpVCwyFyzpCXJ6uJios_Th2cEKEG9Mi6AMs2mzZefNFdole386ospUthTHs" alt="" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" class="lazyload">

<p class="c15">You will be using the dataset<a class="c11" href="https://www.google.com/url?q=http://networkrepository.com/soc-wiki-elec.php&amp;sa=D&amp;ust=1601354266269000&amp;usg=AOvVaw3rOJNLWG9lkNvbKyl-qRty">&nbsp;</a><span class="c25"><a class="c11" href="https://www.google.com/url?q=http://networkrepository.com/soc-wiki-elec.php&amp;sa=D&amp;ust=1601354266269000&amp;usg=AOvVaw3rOJNLWG9lkNvbKyl-qRty">Wikipedia adminship election data</a></span>&nbsp;<span class="c0">which has almost 7K nodes and &nbsp;100K edges. Also, you may find the dataset under the hw4-skeleton/Q1 as “soc-wiki-elec.edges”</span>

<p class="c15">In&nbsp;<span class="c30">pagerank.py</span><span class="c0">,</span>

<ul class="c22 lst-kix_a3r2cfaxj4te-0 start">
<li class="c50"><span class="c0">You have to complete the code to perform the following steps (guidelines are also given in the pagerank.py)</span></li>
</ul>
<ul class="c22 lst-kix_a3r2cfaxj4te-1 start">
<li class="c57"><span class="c30 c33"><span class="c30 c33">Calculate the out-degree of each node and maximum node_id of the graph.</span></span><span class="c17">Maximum node_id refers to the highest value you see for any node id, whether source or target. For eg: &nbsp;if edges are [(1,2),(3,4)] {in the format (source,target)}, the max_node_id here is 4.</span></li>
</ul>
<ul class="c22 lst-kix_wpa8susr8q05-1 start">
<li class="c57"><span class="c17 c33">Store the out-degree in class variable “node_degree” and maximum node id to “max_node_id”</span></li>
</ul>
<ul class="c22 lst-kix_3jjwtumz5xgh-0 start">
<li class="c50">You will be asked to implement the simplified PageRank algorithm, where<span class="c4 c56 c89">&nbsp;</span><span class="c4 c8">Pd ( v</span><span class="c8 c4 c94">j&nbsp;</span><span class="c8 c4">) = 1/n</span><span class="c0"><span class="c0">&nbsp;in the script provided and you are asked to submit the output for 10 and 25 iteration runs.</span></span>To verify, we are providing the sample output of 5 iterations for a simplified pagerank.</li>
</ul>
<ul class="c22 lst-kix_3zt0s3oga689-0 start">
<li class="c50">For personalized PageRank, the&nbsp;<span class="c8 c4">Pd ( )</span><span class="c0">&nbsp;vector will be assigned values based on your 9 digit GTID (Eg: 987654321) and you are asked to submit the output for 10, and 25 iteration runs.</span></li>
<li class="c50"><span class="c0">Syntax to run the code is given in pagerank.py (First line of main function). &nbsp;The given code generates the output file as per the naming convention.</span></li>
</ul>
<p class="c95 c98"><span class="c62 c68 c9 c97">Deliverables:</span>

<ol class="c22 lst-kix_ydz71lrc0mlp-0 start" start="1">
<li class="c70 c52"><span class="c9">pagerank.py [12 pts]</span><span class="c0">: your modified implementation</span></li>
<li class="c70 c52"><span class="c9">simplified_pagerank_{n}.txt</span><span class="c0">: 2 files (as given below) containing the top 10 node IDs and their simplified pageranks for n iterations</span></li>
</ol>
<p class="c32 c20"><span class="c10 c9">simplified_pagerank10.txt [2 pts]</span>

<p class="c32 c20"><span class="c10 c9">simplified_pagerank25.txt [2 pts]</span>

<ol class="c22 lst-kix_rxmt2k6qjced-0 start" start="3">
<li class="c52 c70"><span class="c9">personalized_pagerank_{n}.txt:&nbsp;</span><span class="c0">2 files (as given below) containing the top 10 node IDs and their personalized pageranks for n iterations</span></li>
</ol>
<p class="c32 c20"><span class="c10 c9">personalized_pagerank10.txt [2 pts]</span>

<p class="c32 c20"><span class="c9">personalized_pagerank25.txt [2 pts]</span>

<h2 id="h.1fob9te" class="c42 c93"><span class="c21">Q2 [50 pts] Random Forest Classifier</span></h2>
<h3 id="h.3znysh7" class="c42 c66"><span class="c62 c75 c81 c56">Q2.1 – Random Forest Setup [45 pts]</span></h3>
<p class="c76"><span class="c2 c44">Note: You must use Python 3.7.x for this question.</span>

<p class="c35">You will implement a random forest classifier in Python. The performance of the classifier will be evaluated&nbsp;<span class="c33">via the&nbsp;</span><span class="c33">out-of-bag (OOB) error estimate</span><span class="c33">,</span>&nbsp;using the&nbsp;provided dataset<span class="c0">.</span>

<p class="c35"><span class="c2">Note:</span><span class="c9">&nbsp;</span><span class="c23 c33">You may only use the modules and libraries provided at the top of the .py files included in the skeleton for Q2 and modules from the Python Standard Library. Python wrappers (or modules) may NOT be used for this assignment. Pandas may NOT be used — while we understand that they are useful libraries to learn, completing this question is not critically dependent on their functionality. In addition, to make grading more manageable and to enable our TAs to provide better, more consistent support to our students, we have decided to restrict the libraries accordingly.</span><span class="c23 c53">&nbsp;</span>

<p class="c35"><span class="c53 c23">&nbsp;</span>

<p class="c35">The dataset you will use is the&nbsp;<span class="c7"><a class="c11" href="https://www.google.com/url?q=https://www.kaggle.com/shrutimechlearn/churn-modelling&amp;sa=D&amp;ust=1601354266272000&amp;usg=AOvVaw3Zvz_PGFyUex0Q0ZztBYFO">Churn prediction</a></span>&nbsp;dataset. Each record consists of different attributes of a bank’s customer. The dataset has been pre-processed and cleaned to remove missing attributes. The data is stored in a comma-separated file (csv) in your Q2 folder as&nbsp;<span class="c9">Churn.csv.&nbsp;</span>Each line describes a customer using 10 columns: the first 9 columns represent the attributes of the customer, and the last column is the label&nbsp;<span class="c0">(1 means customer left the bank).</span>

<p class="c35">The original data was unbalanced and random undersampling has been performed to balance the data. You can read more about undersampling&nbsp;<span class="c7"><a class="c11" href="https://www.google.com/url?q=https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis&amp;sa=D&amp;ust=1601354266273000&amp;usg=AOvVaw1D5_zbCknid_KgaG-09pf1">here</a></span><span class="c0">.</span>

<p class="c35"><span class="c2">Note 1:</span><span class="c23">&nbsp;The last column&nbsp;</span><span class="c2">should not</span><span class="c53 c23">&nbsp;be treated as an attribute.</span>

<p class="c35"><span class="c2">Note 2</span><span class="c53 c23">: Do not modify the dataset.</span>

<p class="c35">You will perform binary classification on the dataset to determine if a customer is going to leave the bank or not.

<h3 id="h.2et92p0" class="c45"><span class="c26 c9">Essential Reading</span></h3>
<h5 id="h.tyjcwt" class="c37"><span class="c53 c69">Decision Trees</span></h5>
<p class="c35">To complete this question, you need to develop a good understanding of how decision trees work. We recommend you review the lecture on the decision tree. Specifically, you need to know how to construct decision trees using&nbsp;<span class="c4">Entropy&nbsp;</span>and<span class="c4">&nbsp;Information Gain</span>&nbsp;to select the splitting attribute and split point for the selected attribute. These&nbsp;<span class="c7"><a class="c11" href="https://www.google.com/url?q=http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf&amp;sa=D&amp;ust=1601354266274000&amp;usg=AOvVaw1S64Sw0f0bk0cstDGQHLt4">slides from CMU</a></span>&nbsp;(also mentioned in the lecture) provide an excellent example of how to construct a decision tree using&nbsp;<span class="c4">Entropy</span>&nbsp;and&nbsp;<span class="c4">Information Gain</span><span class="c0">.</span>

<h5 id="h.3dy6vkm" class="c37"><span class="c53 c69">Random Forests</span></h5>
<p class="c35">To refresh your memory about random forests, &nbsp;see Chapter 15 in the&nbsp;<span class="c7"><a class="c11" href="https://www.google.com/url?q=https://web.stanford.edu/~hastie/Papers/ESLII.pdf&amp;sa=D&amp;ust=1601354266274000&amp;usg=AOvVaw3nPM4D_7WmAvDELBJO_Dfx">Elements of Statistical Learning</a></span>&nbsp;book and the lecture on random forests. Here is a&nbsp;<span class="c7"><a class="c11" href="https://www.google.com/url?q=http://blog.echen.me/2011/03/14/laymans-introduction-to-random-forests/&amp;sa=D&amp;ust=1601354266275000&amp;usg=AOvVaw35I5tPfe-XR5VjDxhRDH3x">blog post</a></span><span class="c0">&nbsp;that introduces random forests in a fun way, in layman’s terms.</span>

<h5 id="h.1t3h5sf" class="c37">Out-of-Bag Error Estimate</h5>
<h5 id="h.4d34og8" class="c35 c42"><span class="c33 c60">In random forests, it is not necessary to perform explicit cross-validation or use a separate test set for performance evaluation. Out-of-bag (OOB)</span><span class="c60">&nbsp;error estimate has shown to be reasonably accurate and unbiased. Below, we summarize the key points about OOB described in the</span>&nbsp;<span class="c7"><a class="c11" href="https://www.google.com/url?q=https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm%23ooberr&amp;sa=D&amp;ust=1601354266275000&amp;usg=AOvVaw2AfEY6D88cKbOl_GUeUZO_">original article by Breiman and Cutler</a></span>.</h5>
<p class="c46"><span class="c33">Each tree in the forest is constructed using a different bootstrap sample from the original data. Each bootstrap sample is constructed by randomly sampling from the original dataset&nbsp;</span><span class="c9 c33">with replacement&nbsp;</span><span class="c33">(usually, a bootstrap sample has the</span><span class="c63 c33">&nbsp;</span><span class="c7 c33"><a class="c11" href="https://www.google.com/url?q=http://stats.stackexchange.com/questions/24330/is-there-a-formula-or-rule-for-determining-the-correct-sampsize-for-a-randomfore&amp;sa=D&amp;ust=1601354266276000&amp;usg=AOvVaw0hpy_t5eIxfjIUCzeBn5bO">same size</a></span><span class="c63 c33">&nbsp;</span><span class="c33">as the original dataset). Statistically, about one-third of the cases are left out of the bootstrap sample and not used in the construction of the&nbsp;</span><span class="c4 c33">kth</span><span class="c33">&nbsp;tree. For each record left out in the construction of the&nbsp;</span><span class="c4 c33">kth</span><span class="c33">&nbsp;tree, it can be assigned a class by the&nbsp;</span><span class="c4 c33">kth</span><span class="c0 c33">&nbsp;tree. As a result, each record will have a “test set” classification by the subset of trees that treat the record as an out-of-bag sample. The majority vote for that record will be its predicted class. The proportion of times that a predicted class is not equal to the true class of a record averaged overall records is the OOB error estimate.</span>

<p class="c46"><span class="c33">While splitting a tree at a node,&nbsp;</span><span class="c33">make sure to randomly select a subset of variable</span><span class="c0 c33">&nbsp;(You can use square root of number of attributes) and pick the best variable and split-point among the subset only. This is an important difference between random forest and bagging decision trees.</span>

<p class="c35"><span class="c26 c9">Starter Code</span>

<p class="c46"><span class="c0">We have prepared a starter code written in Python for you to use. This would help you load the data and evaluate your model. The following files are provided for you:</span>

<ul class="c22 lst-kix_vfhjqso34qdd-0 start">
<li class="c1"><span class="c30">util.py</span><span class="c0">: utility functions that will help you build a decision tree</span></li>
<li class="c1"><span class="c30">decision_tree.py</span><span class="c0">: a decision tree class that you will use to build your random forest</span></li>
<li class="c1"><span class="c30">random_forest.py</span><span class="c0">: a random forest class and the main method to test your random forest</span></li>
</ul>
<h3 id="h.2s8eyo1" class="c45"><span class="c6 c9 c69">What you will implement</span></h3>
<p class="c35">Below, we have summarized what you will implement to solve this question. Note that you MUST use&nbsp;<span class="c9">information gain</span><span class="c0">&nbsp;to perform the splitting in the decision tree. The starter code has detailed comments on how to implement each function.</span>

<ol class="c22 lst-kix_92l5gclbjqvp-0 start" start="1">
<li class="c1"><span class="c30">util.py</span><span class="c0">: implement the functions to compute entropy, information gain, perform splitting, &nbsp;and find the best variable &amp; split-point.</span></li>
<li class="c1"><span class="c30">decision_tree.py</span>: implement the&nbsp;<span class="c30">learn()</span>&nbsp;method to build your decision tree using the utility functions above.</li>
<li class="c1"><span class="c30">decision_tree.py</span>: implement the&nbsp;<span class="c30">classify()</span><span class="c4">&nbsp;</span><span class="c0">method to predict the label of a test record using your decision tree.</span></li>
<li class="c1"><span class="c30">random_forest.py</span>: implement the methods&nbsp;<span class="c30">_bootstrapping()</span>,&nbsp;<span class="c30">fitting()</span>,&nbsp;<span class="c30">voting()&nbsp;</span>and<span class="c17">&nbsp;user().</span></li>
</ol>
<p class="c35"><span class="c2">Note 1</span><span class="c23">:</span>&nbsp;<span class="c53 c23">You must achieve a minimum accuracy of 70% for the random forest.</span>

<p class="c35"><span class="c2">Note 2</span><span class="c23">:</span>&nbsp;<span class="c53 c23">Your code must take no more than 5 minutes to execute.</span>

<p class="c35"><span class="c27 c2">Note 3</span><span class="c27 c23 c56">: Remember to remove all of your print statements from the code. Nothing other than the existing print statements in&nbsp;</span><span class="c27 c2">main()</span><span class="c27 c23 c56">&nbsp;should be printed on the console. Failure to do so may result in point deduction. Do not remove the existing print statements in the&nbsp;</span><span class="c2 c27">main()</span><span class="c27 c23 c56">&nbsp;in&nbsp;</span><span class="c27 c2">random_forest.py</span><span class="c27 c23 c56">.</span>

<p class="c35"><span class="c0">As you solve this question, you will need to think about multiple parameters in your design, some may be more straightforward to determine, some maybe not (hint: study lecture slides and essential reading above). For example:</span>

<ul class="c22 lst-kix_hdd75jjaem6m-0 start">
<li class="c13"><span class="c0">Which attributes to use when building a tree?</span></li>
<li class="c13"><span class="c0">How to determine the split point for an attribute?</span></li>
<li class="c13"><span class="c0">When do you stop splitting leaf nodes?</span></li>
<li class="c13"><span class="c0">How many trees should the forest contain?</span></li>
</ul>
<p class="c46">Note that, as mentioned in the lecture, there are other approaches to implement random forests. For example, instead of information gain, other popular choices include the Gini index, random attribute selection (e.g.,&nbsp;<span class="c7"><a class="c11" href="https://www.google.com/url?q=http://citeseerx.ist.psu.edu/viewdoc/download?doi%3D10.1.1.232.2940%26rep%3Drep1%26type%3Dpdf&amp;sa=D&amp;ust=1601354266279000&amp;usg=AOvVaw0PakBfjQLgOH0tNNhRfQpY">PERT – Perfect Random Tree Ensembles</a></span><span class="c0">). We decided to ask everyone to use an information gain based approach in this question (instead of leaving it open-ended), to help standardize students’ solutions to help accelerate our grading efforts.</span>

<h3 id="h.17dp8vu" class="c66 c42"><span class="c62 c75 c56 c81">Q2.2 – forest.txt [5 pts]</span></h3>
<p class="c46">In&nbsp;<span class="c9">forest.txt</span><span class="c0">, report the following:</span>

<ol class="c22 lst-kix_ux5ed6kigvvo-0 start" start="1">
<li class="c46 c52"><span class="c0">What is the main reason to use a random forest versus a decision tree? (&lt;= 50 words)</span></li>
<li class="c46 c52"><span class="c0">How long did your random forest take to run? (in seconds)</span></li>
<li class="c46 c52"><span class="c0">What accuracy (to two decimal places, xx.xx%) were you able to obtain?</span></li>
</ol>
<p class="c46"><span class="c9 c26">Deliverables</span>

<ol class="c22 lst-kix_wp3cl0fjzd7v-0 start" start="1">
<li class="c46 c52"><span class="c9">util.py [10 pts]</span><span class="c0">: The source code of your utility functions.</span></li>
<li class="c46 c52"><span class="c9">decision_tree.py [10 pts]</span><span class="c0">: The source code of your decision tree implementation.</span></li>
<li class="c46 c52"><span class="c9">random_forest.py [25 pts]</span><span class="c0">: The source code of your random forest implementation with appropriate comments.</span></li>
<li class="c46 c52"><span class="c9">forest.txt [5 pts]</span><span class="c0">: The text file containing your responses to Q2.2</span></li>
</ol>
<h2 id="h.2f44k1n77njh" class="c42 c88"><span class="c21">Q3 [30 points] Using Scikit-Learn</span></h2>
<p class="c15"><span class="c9 c28 c44">Note: You must use Python 3.7.x for this q</span><span class="c6 c9 c28">uestion Scikit-Learn v0.22 for this question.</span>

<p class="c15"><span class="c7"><a class="c11" href="https://www.google.com/url?q=http://scikit-learn.org&amp;sa=D&amp;ust=1601354266281000&amp;usg=AOvVaw3tBFfUAfeSpB0pPVaFx6cj">Scikit-learn</a></span><span class="c0">&nbsp;is a popular Python library for machine learning. You will use it to train some classifiers on the Predicting a Pulsar Star dataset which is provided in the hw4-skeleton/Q3 as pulsar_star.csv</span>

<p class="c15"><span class="c19">Note: Your code must take no more than 15 minutes to execute all cells.</span>

<p class="c15"><span class="c0">—————————————————————————————————————————-</span>

<p class="c15">For this problem you will be utilizing and submitting a<span class="c7"><a class="c11" href="https://www.google.com/url?q=https://jupyter.readthedocs.io/en/latest/install.html&amp;sa=D&amp;ust=1601354266281000&amp;usg=AOvVaw1ysTNEYjnCcwlemFkdkwnU">&nbsp;Jupyter notebook</a></span><span class="c0">.</span>

<p class="c15"><span class="c0">For any values we ask you to report in this question, please make sure to print them out in your Jupyter notebook such that they are outputted when we run your code. We’ve included the places in the notebook where a print statement is required.</span>

<p class="c15"><span class="c19">Note: Do not add any additional print statements to the notebook, you may add them for debugging, but please make sure to remove any print statements that aren’t required.</span>

<p class="c15"><span class="c0">—————————————————————————————————————————-</span>

<h4 id="h.qqgu9vya8qi" class="c39"><span class="c6 c40">Q3.1 – Data Import and Cleansing Setup</span></h4>
<p class="c15 c87">In this step you will import the pulsar data set and allocate the data to two separate arrays. Once this is completed, you will then split the data into a training and test set using the scikit-learn function&nbsp;<span class="c7"><a class="c11" href="https://www.google.com/url?q=https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html&amp;sa=D&amp;ust=1601354266282000&amp;usg=AOvVaw03Ijmlrpn8giT4Pf6ZWvsW">train_test_split</a></span><span class="c0">. In the proceeding steps you will use scikit-learn’s built in machine learning algorithms to predict accuracy of each from the given dataset. Each algorithm has additional documentation (which we’ve provided in the links) explaining them in more detail, such as how they work and how to use them.</span>

<h4 id="h.e83kkey0jixd" class="c39"><span class="c6 c40">Q3.2 – Linear Regression Classifier [4 pts]</span></h4>
<p class="c15"><span class="c0">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Q3.2.1 – Classification</span>

<p class="c15">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="c7"><a class="c11" href="https://www.google.com/url?q=https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html&amp;sa=D&amp;ust=1601354266283000&amp;usg=AOvVaw0KmJKqUDav3o5qAelJz0VY">Linear Regression</a></span>&nbsp;–&nbsp;<span class="c0">Train the following classifier on the dataset, using the class provided in the link. You will need to provide accuracy on both the test and train sets. The challenge here will be making sure that you round your predictions to a binary 0 or 1. See the jupyter notebook for more information.</span>

<h4 id="h.51l2wlkmpaw4" class="c39"><span class="c6 c40">Q3.3 – Random Forest Classifier [10 pts]</span></h4>
<p class="c15"><span class="c0">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Q3.3.1 – Classification &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>

<p class="c15">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="c7"><a class="c11" href="https://www.google.com/url?q=http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html&amp;sa=D&amp;ust=1601354266283000&amp;usg=AOvVaw2tP7H4xgTxRCRTl0Snhksr">Random Forest</a></span><span class="c0">&nbsp;– Train the following classifier on the dataset, using the class provided in the links You will need to provide accuracy on both the test and train sets. You will not be required to round your prediction in this section.</span>

<p class="c15"><span class="c0">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Q3.3.2 – Feature Importance</span>

<p class="c15 c99"><span class="c0">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;You have performed a simple classification task using the random forest algorithm. You have also implemented the algorithm in Q2 above. The concept of entropy gain can also be used to evaluate the importance of a feature. In this section you will determine the feature importance as evaluated by the random forest Classifier. You must then sort them in descending order and print the feature numbers. Hint: There is a direct function available in sklearn to achieve this. Also checkout argsort() function in Python numpy. (argsort() returns the indices of the elements in ascending order) You should use the first classifier that you trained initially in Q3.1, without any kind of hyperparameter-tuning, for reporting these features.</span>

<p class="c15"><span class="c0">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Q3.3.3 – Hyper-Parameter Tuning</span>

<p class="c15">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Tune your random forest&nbsp;to obtain their best accuracy on the dataset. For random forest, tune the model on the unmodified test and train datasets. &nbsp;Tune the hyperparameters specified below, using the<span class="c7"><a class="c11" href="https://www.google.com/url?q=http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html&amp;sa=D&amp;ust=1601354266284000&amp;usg=AOvVaw2UITdOw0rB1ORk7xZG3t7O">&nbsp;GridSearchCV</a></span><span class="c0">&nbsp;function that Scikit provides:</span>

<p class="c48"><span class="c9">&nbsp;</span><span class="c9">‘n_estimators’: [4, 16, 256], ’max_depth’: [2, 8, 16]</span>

<h4 id="h.2axpcmcv8160" class="c39"><span class="c6 c40">Q3.4 – Support Vector Machine [12 pts]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span></h4>
<p class="c15"><span class="c0">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Q3.4.1 – Preprocessing</span>

<p class="c15">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For SVM, we will&nbsp;<span class="c7"><a class="c11" href="https://www.google.com/url?q=http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html&amp;sa=D&amp;ust=1601354266285000&amp;usg=AOvVaw3LFNeE-WNR0D1ULF34b8F7">standardize</a></span>&nbsp;attribu<span class="c0">tes (features) in the dataset before using it to tune the model.</span>

<p class="c15">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="c9">Note:</span><span class="c0">&nbsp;</span>

<p class="c15"><span class="c0">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For StandardScaler:</span>

<ul class="c22 lst-kix_6bp9dc7u6hfp-1 start">
<li class="c15 c61">Pass x_train into the fit method.&nbsp;Then transform both x_train and x_test to obtain the standardized versions of both.</li>
<li class="c15 c61"><span class="c0">The reason we fit only on x_train and not the entire dataset is because we do not want to train on data that was affected by the testing set.</span></li>
<li class="c15 c61"><span class="c0">Please see the link above for information and implementation instructions.</span></li>
</ul>
<p class="c15"><span class="c0">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Q3.4.2 – Classification</span>

<p class="c15">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="c7"><a class="c11" href="https://www.google.com/url?q=https://scikit-learn.org/stable/modules/svm.html&amp;sa=D&amp;ust=1601354266286000&amp;usg=AOvVaw2haobzzQCZTfLmqSfodN-q">Support Vector Machine</a></span>&nbsp;(T<span class="c0">he link points to SVC, which is a particular implementation of SVM by Scikit.) Train the following classifier on the dataset, using the classes provided in the links. You will need to provide accuracy on both the test and train sets.</span>

<p class="c15"><span class="c0">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Q3.4.3. – &nbsp;Hyper-Parameter Tuning</span>

<p class="c15">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Tune your SVM model to obtain their best accuracy on the dataset. For SVM, tune the model on the&nbsp;standardized test and train datasets.&nbsp;Tune the hyperparameters specified below, using the<span class="c7"><a class="c11" href="https://www.google.com/url?q=http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html&amp;sa=D&amp;ust=1601354266287000&amp;usg=AOvVaw0BEf6I5d9prDfoWLyrui5B">&nbsp;GridSearchCV</a></span><span class="c0">&nbsp;function that Scikit provides:</span>

<p class="c48"><span class="c9">‘kernel’:(‘linear’, ‘rbf’), ‘C’:[0.01, 0.1, 1.0]</span>

<p class="c15"><span class="c0">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>

<p class="c15">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="c19">Note: If GridSearchCV is taking a long time to run for SVM, make sure you are standardizing your data beforehand.</span>

<p class="c15"><span class="c0">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>

<p class="c15"><span class="c0">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Q3.4.4. – Cross-Validation Results</span>

<p class="c15"><span class="c0">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Let’s practice getting the results of cross-validation. For your SVM (only), report the rank test score and &nbsp;mean testing score for the best combination of hyper-parameter values that you obtained. The GridSearchCV class holds a ‘cv_results_’ dictionary that should help you report these metrics easily.</span>

<p class="c15"><span class="c0">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>

<h4 id="h.siti8vknarhq" class="c39"><span class="c6 c40">Q3.5 – Principal Component Analysis [4 pts]</span></h4>
<p class="c15">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="c7"><a class="c11" href="https://www.google.com/url?q=https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html&amp;sa=D&amp;ust=1601354266288000&amp;usg=AOvVaw0M0IW9zCDafbemBKGOZZ6H">Principal Component Analysis</a></span>&nbsp;Dimensionality reduction is an important task in many data analysis exercises and it involves projecting the data to a lower dimensional space using Singular Value Decomposition. Refer to the examples given<span class="c7"><a class="c11" href="https://www.google.com/url?q=https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html&amp;sa=D&amp;ust=1601354266288000&amp;usg=AOvVaw0M0IW9zCDafbemBKGOZZ6H">&nbsp;here</a></span><span class="c0">, set parameters n_component to 8 and svd_solver to ‘full’. See the sample outputs below.</span>

<p class="c15"><span class="c19">Note: For this section use the unmodified data(x_data)</span>

<ol class="c22 lst-kix_9nkpijwvvngm-0 start" start="1">
<li class="c15 c52"><span class="c0">Percentage of variance explained by each of the selected components.</span></li>
</ol>
<p class="c48"><span class="c10 c9">Sample Output:</span>

<p class="c48"><span class="c10 c9">[6.51153033e-01 5.21914311e-02 2.11562330e-02 5.15967655e-03</span>

<p class="c48"><span class="c10 c9">&nbsp;6.23717966e-03 4.43578490e-04 9.77570944e-05 7.87968645e-06]</span>

<ol class="c22 lst-kix_da0mpdtd2btr-0 start" start="2">
<li class="c15 c52"><span class="c0">The singular values corresponding to each of the selected components.</span></li>
</ol>
<p class="c48"><span class="c10 c9">Sample Output:</span>

<p class="c48"><span class="c10 c9">[5673.123456 &nbsp;4532.123456 &nbsp; 4321.68022725 &nbsp;1500.47665361</span>

<p class="c48"><span class="c10 c9">&nbsp; &nbsp;1250.123456 &nbsp; 750.123456 &nbsp; &nbsp;100.123456 &nbsp; &nbsp;30.123456]</span>

<p class="c15"><span class="c9 c28">Use the jupyter notebook skeleton file called</span><span class="c9 c28 c4">&nbsp;</span><span class="c9 c28">hw4q3.ipynb</span><span class="c9 c4 c28">&nbsp;</span><span class="c62 c9 c28 c68">to write and execute your code.</span>

<p class="c15"><span class="c54 c4">As a reminder, the general flow of your machine learning code will look like:</span>

<ol class="c22 lst-kix_8yh70ardzb7-0 start" start="1">
<li class="c50"><span class="c4 c54">Load dataset</span></li>
<li class="c50"><span class="c54 c4">Preprocess (you will do this in Q3.2)</span></li>
<li class="c50"><span class="c4">Split the data into&nbsp;</span><span class="c30 c4">x_train</span><span class="c4">,&nbsp;</span><span class="c30 c4">y_train</span><span class="c4">,&nbsp;</span><span class="c30 c4">x_test</span><span class="c4">,&nbsp;</span><span class="c30 c4">y_test</span><span class="c54 c4">&nbsp;</span></li>
<li class="c50"><span class="c4">Train the classifier on&nbsp;</span><span class="c4 c30">x_train</span><span class="c4">&nbsp;and&nbsp;</span><span class="c78 c30 c4 c60">y_train</span></li>
<li class="c50"><span class="c4">Predict on&nbsp;</span><span class="c30 c4 c60 c78">x_test</span></li>
<li class="c50"><span class="c54 c4">Evaluate testing accuracy by comparing the predictions from step 5 with y_test.</span></li>
</ol>
<p class="c86"><span class="c4">Here is an</span><span class="c4"><a class="c11" href="https://www.google.com/url?q=https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html&amp;sa=D&amp;ust=1601354266290000&amp;usg=AOvVaw3V5HI0vkY2vU44ZrawET6U">&nbsp;</a></span><span class="c25 c4"><a class="c11" href="https://www.google.com/url?q=https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html&amp;sa=D&amp;ust=1601354266290000&amp;usg=AOvVaw3V5HI0vkY2vU44ZrawET6U">example</a></span><span class="c54 c4">. Scikit has many other examples as well that you can learn from.</span>

<p class="c55"><span class="c53 c63">Deliverables</span>

<ul class="c22 lst-kix_muvl2fg044jc-0 start">
<li class="c52 c77"><span class="c9">hw4q3.ipynb&nbsp;</span>– jupyter notebook file filled with your code for part Q3.1-Q3.5.</li>
</ul>
<ul class="c22 lst-kix_oizq23o5j7uj-0 start">
<li class="c95 c52 c96"><span class="c10 c9">Submission Guidelines</span></li>
</ul>
<p class="c15">Submit the deliverables as a single&nbsp;<span class="c9 c92">zip</span>&nbsp;file named&nbsp;<span class="c9">HW4-GTusername.zip</span><span class="c0">. Write down the name(s) of any students you have collaborated with on this assignment, using the text box on the Canvas submission page.</span>

<p class="c15"><span class="c0">The zip file’s directory structure must exactly be (when unzipped):</span>

<p class="c15"><span class="c30">HW4-</span><span class="c30">GTusername</span><span class="c17">/</span>

<p class="c15 c20"><span class="c17">Q1/</span>

<p class="c32 c20"><span class="c17">pagerank.py</span>

<p class="c12"><span class="c17">simplified_pagerank10.txt</span>

<p class="c12"><span class="c17">simplified_pagerank25.txt</span>

<p class="c12"><span class="c17">personalized_pagerank10.txt</span>

<p class="c12"><span class="c17">personalized_pagerank25.txt</span>

<p class="c15 c20"><span class="c17">Q2/</span>

<p class="c20 c32"><span class="c17">util.py</span>

<p class="c32 c20"><span class="c17">decision_tree.py</span>

<p class="c32 c20"><span class="c17">random_forest.py</span>

<p class="c32 c20"><span class="c17">forest.txt</span>

<p class="c15 c20"><span class="c17">Q3/</span>

<p class="c32 c20"><span class="c30">hw4q3.ipynb</span>
