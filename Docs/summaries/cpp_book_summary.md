# C++ Neural Networks and Fuzzy Logic - Chapter Summaries
**By Valluru B. Rao** | ISBN: 1558515526 | 1995

---

## Chapter 1: Introduction to Neural Networks
*(Pages 14-30)*

C++ Neural Networks and Fuzzy Logic
by Valluru B. Rao
MTBooks, IDG Books Worldwide, Inc.
ISBN: 1558515526   Pub Date: 06/01/95

Chapter 1
Introduction to Neural Networks
Neural Processing
How do you recognize a face in a crowd? How does an economist predict the direction of interest rates? Faced
with problems like these, the human brain uses a web of interconnected processing elements called neurons to
process information. Each neuron is autonomous and independent; it does its work asynchronously, that is,
without any synchronization to other events taking place. The two problems posed, namely recognizing a face
and forecasting interest rates, have two important characteristics that distinguish them from other problems:
First, the problems are complex, that is, you can’t devise a simple step−by−step algorithm or precise formula
to give you an answer; and second, the data provided to resolve the problems is equally complex and may be
noisy or incomplete. You could have forgotten your glasses when you’re trying to recognize that face. The
economist may have at his or her disposal thousands of pieces of data that may or may not be relevant to his
or her forecast on the economy and on interest rates.
The vast processing power inherent in biological neural structures has inspired the study of the structure itself
for hints on organizing human−made computing structures. Artificial neural networks, the subject of this
book, covers the way to organize synthetic neurons to solve the same kind of difficult, complex problems in a
similar manner as we think the human brain may. This chapter will give you a sampling of the terms and
nomenclature used to talk about neural networks. These terms will be covered in more depth in the chapters to
follow.
Neural Network
A neural network is a computational structure inspired by the study of biological neural processing. There are
many different types of neural networks, from relatively simple to very complex, just as there are many
theories on how biological neural processing works. We will begin with a discussion of a layered
feed−forward type of neural network and branch out to other paradigms later in this chapter and in other
chapters.
A layered feed−forward neural network has layers, or subgroups of processing elements. A layer of
processing elements makes independent computations on data that it receives and passes the results to another
layer. The next layer may in turn make its independent computations and pass on the results to yet another
layer. Finally, a subgroup of one or more processing elements determines the output from the network. Each
processing element makes its computation based upon a weighted sum of its inputs. The first layer is the input
layer and the last the output layer. The layers that are placed between the first and the last layers are the
hidden layers. The processing elements are seen as units that are similar to the neurons in a human brain, and
hence, they are referred to as cells, neuromimes, or artificial neurons. A threshold function is sometimes used
to qualify the output of a neuron in the output layer. Even though our subject matter deals with artificial
neurons, we will simply refer to them as neurons. Synapses between neurons are referred to as connections,
Chapter 1 Introduction to Neural Networks
14


---

## Chapter 2: C++ and Object Orientation
*(Pages 30-50)*

C++ Neural Networks and Fuzzy Logic
by Valluru B. Rao
MTBooks, IDG Books Worldwide, Inc.
ISBN: 1558515526   Pub Date: 06/01/95

Chapter 2
C++ and Object Orientation
Introduction to C++
C++ is an object−oriented programming language built on the base of the C language. This chapter gives you
a very brief introduction to C++, touching on many important aspects of C++, so you would be able to follow
our presentations of the C++ implementations of neural network models and write your own C++ programs.
The C++ language is a superset of the C language. You could write C++ programs like C programs (a few of
the programs in this book are like that), or you could take advantage of the object−oriented features of C++ to
write object−oriented programs (like the backpropagation simulator of Chapter 7). What makes a
programming language or programming methodology object oriented? Well, there are several indisputable
pillars of object orientation. These features stand out more than any other as far as object orientation goes.
They are encapsulation, data hiding, overloading, polymorphism, and the grand−daddy of them all:
inheritance. Each of the pillars of object−orientation will be discussed in the coming sections, but before we
tackle these, we need to answer the question, What does all this object−oriented stuff buy me ? By using the
object−oriented features of C++, in conjunction with Object−Oriented Analysis and Design(OOAD), which is
a methodology that fully utilizes object orientation, you can have well−packaged, reusable, extensible, and
reliable programs and program segments. It’s beyond the scope of this book to discuss OOAD, but it’s
recommended you read Booch or Rumbaugh to get more details on OOAD and how and why to change your
programming style forever! See the reference section in the back of this book for more information on these
readings. Now let’s get back to discussing the great object−oriented features of C++.
Encapsulation
In C++ you have the facility to encapsulate data and the operations that manipulate that data, in an appropriate
object. This enables the use of these collections of data and function, called objects , in programs other than
the program for which they were originally created. With objects, just as with the traditional concept of
subroutines, you make functional blocks of code. You still have language−supported abstractions such as
scope and separate compilation available. This is a rudimentary form of encapsulation. Objects carry
encapsulation a step further. With objects, you define not only the way a function operates, or its
implementation, but also the way an object can be accessed, or its interface. You can specify access
differently for different entities. For example, you could make function do_operation() contained inside
Object A accessible to Object B but not to Object C. This access qualification can also be used for data
members inside an object. The encapsulation of data and the intended operations on them prevents the data
from being subjected to operations not meant for them. This is what really makes objects reusable and
portable! The operations are usually given in the form of functions operating upon the data items. Such
functions are also called methods in some object−oriented programming languages. The data items and the
functions that manipulate them are combined into a structure called a class. A class is an abstract data type.
When you make an instance of a class, you have an object. This is no different than when you instantiate an
Chapter 2 C++ and Object Orientation
30


---

## Chapter 3: A Look at Fuzzy Logic
*(Pages 50-70)*

12.5
Output fuzzy category is ==> v.tight<==
category   membership
−−−−−−−−−−−−−−−−−−−−−−−−−−−−−
v.accommodative      0
accommodative        0
tight        0
v.tight      1
input a data value, type 0 to terminate
0
All done. Have a fuzzy day !
Fuzzy Control Systems
The most widespread use of fuzzy logic today is in fuzzy control applications. You can use fuzzy logic to
make your air conditioner cool your room. Or you can design a subway system to use fuzzy logic to control
the braking system for smooth and accurate stops. A control system is a closed−loop system that typically
controls a machine to achieve a particular desired response, given a number of environmental inputs. A fuzzy
control system is a closed−loop system that uses the process of fuzzification, as shown in the Federal Reserve
policy program example, to generate fuzzy inputs to an inference engine, which is a knowledge base of
actions to take. The inverse process, called defuzzification, is also used in a fuzzy control system to create
crisp, real values to apply to the machine or process under control. In Japan, fuzzy controllers have been used
to control many machines, including washing machines and camcorders.
Figure 3.3 shows a diagram of a fuzzy control system. The major parts of this closed−loop system are:
Figure 3.3  Diagram of a fuzzy control system.
•
machine under control—this is the machine or process that you are controlling, for example, a
washing machine
•
outputs—these are the measured response behaviors of your machine, for example, the
temperature of the water
•
fuzzy outputs—these are the same outputs passed through a fuzzifier, for example, hot or very
cold
•
inference engine/fuzzy rule base—an inference engine converts fuzzy outputs to actions to take
by accessing fuzzy rules in a fuzzy rule base. An example of a fuzzy rule: IF the output is very cold,
THEN increase the water temperature setting by a very large amount
•
fuzzy inputs—these are the fuzzy actions to perform, such as increase the water temperature
setting by a very large amount
•
inputs—these are the (crisp) dials on the machine to control its behavior, for example, water
temperature setting = 3.423, converted from fuzzy inputs with a defuzzifier
The key to development of a fuzzy control system is to iteratively construct a fuzzy rule base that yields the
desired response from your machine. You construct these fuzzy rules from knowledge about the problem. In
many cases this is very intuitive and gives you a robust control system in a very short amount of time.
Fuzzy Control Systems
50

Copyright © IDG Books Worldwide, Inc.
Fuzzy Control Systems
51

C++ Neural Networks and Fuzzy Logic
by Valluru B. Rao
MTBooks, IDG Books Worldwide, Inc.
ISBN: 1558515526   Pub Date: 06/01/95

Fuzziness in Neural Networks
Fuzziness can enter neural networks to define the weights from fuzzy sets. A comparison between expert
systems and fuzzy systems is important to understand in the context of neural networks. Expert systems are
based on crisp rules. Such crisp rules may not always be available. Expert systems have to consider an
exhaustive set of possibilities. Such sets may not be known beforehand. When crisp rules are not possible, and
when it is not known if the possibilities are exhaustive, the expert systems approach is not a good one.
Some neural networks, through the features of training and learning, can function in the presence of
unexpected situations. Therein neural networks have an advantage over expert systems, and they can manage
with far less information than expert systems need.
One form of fuzziness in neural networks is called a fuzzy cognitive map. A fuzzy cognitive map is like a
dynamic state machine with fuzzy states. A traditional state machine is a machine with defined states and
outputs associated with each state. Transitions from state to state take place according to input events or
stimuli. A fuzzy cognitive map looks like a state machine but has fuzzy states (not just 1 or 0). You have a set
of weights along each transition path, and these weights can be learned from a set of training data.
Our treatment of fuzziness in neural networks is with the discussion of the fuzzy associative memory,
abbreviated as FAM, which, like the fuzzy cognitive map, was developed by Bart Kosko. The FAM and the
C++ implementation are discussed in Chapter 9.
Neural−Trained Fuzzy Systems
So far we have considered how fuzzy logic plays a role in neural networks. The converse relationship, neural
networks in fuzzy systems, is also an active area of research. In order to build a fuzzy system, you must have a
set of membership rules for fuzzy categories. It is sometimes difficult to deduce these membership rules with
a given set of complex data. Why not use a neural network to define the fuzzy rules for you? A neural network
is good at discovering relationships and patterns in data and can be used to preprocess data in a fuzzy system.
Further, a neural network that can learn new relationships with new input data can be used to refine fuzzy
rules to create a fuzzy adaptive system. Neural trained fuzzy systems are being used in many commercial
applications, especially in Japan:
•  The Laboratory for International Fuzzy Engineering Research (LIFE) in Yokohama, Japan has a
backpropagation neural network that derives fuzzy rules and membership functions. The LIFE system
has been successfully applied to a foreign−exchange trade support system with approximately 5000
fuzzy rules.
•  Ford Motor Company has developed trainable fuzzy systems for automobile idle−speed control.
•  National Semiconductor Corporation has a software product called NeuFuz that supports the
generation of fuzzy rules with a neural network for control applications.
•  A number of Japanese consumer and industrial products use neural networks with fuzzy systems,
including vacuum cleaners, rice cookers, washing machines, and photocopying machines.
Fuzziness in Neural Networks
52


---

## Chapter 4: Constructing a Neural Network
*(Pages 70-110)*

{
     cout << " Can't open a file\n";
     exit(1);
     }
cout<<"\nTHIS PROGRAM IS FOR A PERCEPTRON NETWORK WITH AN INPUT LAYER OF";
cout<<"\n4 NEURONS, EACH CONNECTED TO THE OUTPUT NEURON.\n";
cout<<"\nTHIS EXAMPLE TAKES REAL NUMBERS AS INPUT SIGNALS\n";
//create the network by calling its constructor.
//the constructor calls neuron constructor as many times as the number of
//neurons in input layer of the network.
cout<<"please enter the number of weights/vectors \n";
cin >> vecnum;
for (i=1;i<=vecnum;i++)
     {
     fscanf(wfile,"%f %f %f %f\n", &wtv1[0],&wtv1[1],&wtv1[2],&wtv1[3]);
     network h1(wtv1[0],wtv1[1],wtv1[2],wtv1[3]);
     fscanf(infile,"%f %f %f %f \n",
     &inputv1[0],&inputv1[1],&inputv1[2],&inputv1[3]);
     cout<<"this is vector # " << i << "\n";
     cout << "please enter a threshold value, eg 7.0\n";
     cin >> threshold;
     h1.onrn.actvtion(inputv1, h1.nrn);
     h1.onrn.outvalue(threshold);
     cout<<"\n\n";
     }
fclose(wfile);
fclose(infile);
}

Copyright © IDG Books Worldwide, Inc.
Implementation of Functions
70

C++ Neural Networks and Fuzzy Logic
by Valluru B. Rao
MTBooks, IDG Books Worldwide, Inc.
ISBN: 1558515526   Pub Date: 06/01/95

Comments on Your C++ Program
Notice the use of input stream operator cin>> in the C++ program, instead of the C function scanf in several
places. The iostream class in C++ was discussed earlier in this chapter. The program works like this:
First, the network input neurons are given their connection weights, and then an input vector is presented to
the input layer. A threshold value is specified, and the output neuron does the weighted sum of its inputs,
which are the outputs of the input layer neurons. This weighted sum is the activation of the output neuron, and
it is compared with the threshold value, and the output neuron fires (output is 1) if the threshold value is not
greater than its activation. It does not fire (output is 0) if its activation is smaller than the threshold value. In
this implementation, neither supervised nor unsupervised training is incorporated.
Input/Output for percept.cpp
There are two data files used in this program. One is for setting up the weights, and the other for setting up the
input vectors. On the command line, you enter the program name followed by the weight file name and the
input file name. For this discussion (also on the accompanying disk for this book) create a file called
weight.dat, which contains the following data:
  2.0 3.0 3.0 2.0
  3.0 0.0 6.0 2.0
These are two weight vectors. Create also an input file called input.dat with the two data vectors below:
  1.95 0.27 0.69 1.25
  0.30 1.05 0.75 0.19
During the execution of the program, you are first prompted for the number of vectors that are used (in this
case, 2), then for a threshold value for the input/weight vectors (use 7.0 in both cases). You will then see the
following output. Note that the user input is in italic.
  percept weight.dat input.dat
THIS PROGRAM IS FOR A PERCEPTRON NETWORK WITH AN INPUT LAYER OF 4
NEURONS, EACH CONNECTED TO THE OUTPUT NEURON.
THIS EXAMPLE TAKES REAL NUMBERS AS INPUT SIGNALS
please enter the number of weights/vectors
2
this is vector # 1
please enter a threshold value, eg 7.0
7.0
weight for neuron 1 is  2           activation is 3.9
weight for neuron 2 is  3           activation is 0.81
Comments on Your C++ Program
71


---

## Chapter 5: A Survey of Neural Network Models
*(Pages 110-140)*

C++ Neural Networks and Fuzzy Logic
by Valluru B. Rao
MTBooks, IDG Books Worldwide, Inc.
ISBN: 1558515526   Pub Date: 06/01/95

Lyapunov Function
Neural networks are dynamic systems in the learning and training phase of their operation, and convergence is
an essential feature, so it was necessary for the researchers developing the models and their learning
algorithms to find a provable criterion for convergence in a dynamic system. The Lyapunov function,
mentioned previously, turned out to be the most convenient and appropriate function. It is also referred to as
the energy function. The function decreases as the system states change. Such a function needs to be found
and watched as the network operation continues from cycle to cycle. Usually it involves a quadratic form. The
least mean squared error is an example of such a function. Lyapunov function usage assures a system stability
that cannot occur without convergence. It is convenient to have one value, that of the Lyapunov function
specifying the system behavior. For example, in the Hopfield network, the energy function is a constant times
the sum of products of outputs of different neurons and the connection weight between them. Since pairs of
neuron outputs are multiplied in each term, the entire expression is a quadratic form.
Other Training Issues
Besides the applications for which a neural network is intended, and depending on these applications, you
need to know certain aspects of the model. The length of encoding time and the length of learning time are
among the important considerations. These times could be long but should not be prohibitive. It is important
to understand how the network behaves with new inputs; some networks may need to be trained all over
again, but some tolerance for distortion in input patterns is desirable, where relevant. Restrictions on the
format of inputs should be known.
An advantage of neural networks is that they can deal with nonlinear functions better than traditional
algorithms can. The ability to store a number of patterns, or needing more and more neurons in the output
field with an increasing number of input patterns are the kind of aspects addressing the capabilities of a
network and also its limitations.
Adaptation
Sometimes neural networks are used as adaptive filters, the motivation for such an architecture being
selectivity. You want the neural network to classify each input pattern into its appropriate category. Adaptive
models involve changing of connection weights during all their operations, while nonadaptive ones do not
alter the weights after the phase of learning with exemplars. The Hopfield network is often used in modeling a
neural network for optimization problems, and the Backpropagation model is a popular choice in most other
applications. Neural network models are distinguishable sometimes by their architecture, sometimes by their
adaptive methods, and sometimes both. Methods for adaptation, where adaptation is incorporated, assume
great significance in the description and utility of a neural network model.
For adaptation, you can modify parameters in an architecture during training, such as the learning rate in the
backpropagation training method for example. A more radical approach is to modify the architecture itself
during training. New neural network paradigms change the number or layers and the number of neurons in a
Lyapunov Function
110


---

## Chapter 6: Learning and Training
*(Pages 140-165)*

}
fprintf(outfile,"\n−−−−−−−−−−\n");
}
void network::list_errors()
{
int i;
for (i=1; i<number_of_layers; i++)
       {
       cout << "layer number : " <<i<< "\n";
       ((output_layer *)layer_ptr[i])
              −>list_errors();
       }
}
int network::fill_IObuffer(FILE * inputfile)
{
// this routine fills memory with
// an array of input, output vectors
// up to a maximum capacity of
// MAX_INPUT_VECTORS_IN_ARRAY
// the return value is the number of read
// vectors
int i, k, count, veclength;
int ins, outs;
ins=layer_ptr[0]−>num_outputs;
outs=layer_ptr[number_of_layers−1]−>num_outputs;
if (training==1)
       veclength=ins+outs;
else
       veclength=ins;
count=0;
while  ((count<MAX_VECTORS)&&
              (!feof(inputfile)))
       {
       k=count*(veclength);
       for (i=0; i<veclength; i++)
              {
              fscanf(inputfile,"%f",&buffer[k+i]);
              }
       fscanf(inputfile,"\n");
       count++;
       }
if (!(ferror(inputfile)))
       return count;
else return −1; // error condition
}
void network::set_up_pattern(int buffer_index)
{
C++ Classes and Class Hierarchy
140

// read one vector into the network
int i, k;
int ins, outs;
ins=layer_ptr[0]−>num_outputs;
outs=layer_ptr[number_of_layers−1]−>num_outputs;
if (training==1)
       k=buffer_index*(ins+outs);
else
       k=buffer_index*ins;
for (i=0; i<ins; i++)
       layer_ptr[0]−>outputs[i]=buffer[k+i];
if (training==1)
{
       for (i=0; i<outs; i++)
              ((output_layer *)layer_ptr[number_of_layers−1])−>
                     expected_values[i]=buffer[k+i+ins];
}
}
void network::forward_prop()
{
int i;
for (i=0; i<number_of_layers; i++)
       {
       layer_ptr[i]−>calc_out(); //polymorphic
                                 // function
       }
}
void network::backward_prop(float & toterror)
{
int i;
// error for the output layer
((output_layer*)layer_ptr[number_of_layers−1])−>
                      calc_error(toterror);
// error for the middle layer(s)
for (i=number_of_layers−2; i>0; i−−)
       {
       ((middle_layer*)layer_ptr[i])−>
                     calc_error();
       }
}

Copyright © IDG Books Worldwide, Inc.
C++ Classes and Class Hierarchy
141

C++ Neural Networks and Fuzzy Logic
by Valluru B. Rao
MTBooks, IDG Books Worldwide, Inc.
ISBN: 1558515526   Pub Date: 06/01/95

A Look at the Functions in the layer.cpp File
The following is a listing of the functions in the layer.cpp file along with a brief statement of each one's
purpose.
•
void set_training(const unsigned &) Sets the value of the private data member, training; use 1
for training mode, and 0 for test mode.
•
unsigned get_training_value() Gets the value of the training constant that gives the mode in use.
•
void get_layer_info() Gets information about the number of layers and layer sizes from the user.
•
void set_up_network() This routine sets up the connections between layers by assigning pointers
appropriately.
•
void randomize_weights() At the beginning of the training process, this routine is used to
randomize all of the weights in the network.
•
void update_weights(const float) As part of training, weights are updated according to the
learning law used in backpropagation.
•
void write_weights(FILE *) This routine is used to write weights to a file.
•
void read_weights(FILE *) This routine is used to read weights into the network from a file.
•
void list_weights() This routine can be used to list weights while a simulation is in progress.
•
void write_outputs(FILE *) This routine writes the outputs of the network to a file.
•
void list_outputs() This routine can be used to list the outputs of the network while a simulation
is in progress.
•
void list_errors() Lists errors for all layers while a simulation is in progress.
•
void forward_prop() Performs the forward propagation.
•
void backward_prop(float &) Performs the backward error propagation.
•
int fill_IObuffer(FILE *) This routine fills the internal IO buffer with data from the training or
test data sets.
•
void set_up_pattern(int) This routine is used to set up one pattern from the IO buffer for
training.
•
inline float squash(float input) This function performs the sigmoid function.
•
inline float randomweight (unsigned unit) This routine returns a random weight between –1 and
1; use 1 to initialize the generator, and 0 for all subsequent calls.
Note that the functions squash(float) and randomweight(unsigned) are declared inline. This
means that the function's source code is inserted wherever it appears. This increases code size,
but also increases speed because a function call, which is expensive, is avoided.
The final file to look at is the backprop.cpp file presented in Listing 7.3.
Listing 7.3 The backprop.cpp file for the backpropagation simulator
// backprop.cpp         V. Rao, H. Rao
C++ Classes and Class Hierarchy
142


---

## Chapter 7: Backpropagation
*(Pages 165-220)*

}
}
void network::compr1(int j,int k)
{
int i;
for(i=0;i<anmbr;++i){
       if(pp[j].v1[i] != pp[k].v1[i]) flag = 1;
       break;
       }
}
void network::compr2(int j,int k)
{
int i;
for(i=0;i<anmbr;++i){
       if(pp[j].v2[i] != pp[k].v2[i]) flag = 1;
       break;}
}
void network::comput1()
{
int j;
for(j=0;j<bnmbr;++j){
       int ii1;
       int c1 =0,d1;
       cout<<”\n”;
       for(ii1=0;ii1<anmbr;++ii1){
              d1 = outs1[ii1] * mtrx1[ii1][j];
              c1 += d1;
              }
       bnrn[j].activation = c1;
       cout<<”\n output layer neuron         “<<j<<” activation is”
              <<c1<<”\n”;
if(bnrn[j].activation <0) {
       bnrn[j].output = 0;
       outs2[j] = 0;}
else
       if(bnrn[j].activation>0) {
               bnrn[j].output = 1;
               outs2[j] = 1;}
               else
               {cout<<”\n A 0 is obtained, use previous output
value \n”;
               if(ninpt<=nexmplr){
                         bnrn[j].output = e[ninpt−1].v2[j];}
Source File
165

else
                         { bnrn[j].output = pp[0].v2[j];}
                         outs2[j] = bnrn[j].output; }
       cout<<”\n output layer neuron         “<<j<<” output is”
               <<bnrn[j].output<<”\n”;
       }
}
void network::comput2()
{
int i;
for(i=0;i<anmbr;++i){
       int ii1;
       int c1=0;
       for(ii1=0;ii1<bnmbr;++ii1){
              c1 += outs2[ii1] * mtrx2[ii1][i];  }
       anrn[i].activation = c1;
       cout<<”\ninput layer neuron        “<<i<<”activation is “
              <<c1<<”\n”;
       if(anrn[i].activation <0 ){
              anrn[i].output = 0;
              outs1[i] = 0;}
       else
              if(anrn[i].activation >0 ) {
                     anrn[i].output = 1;
                     outs1[i] = 1;
                     }
              else
              { cout<<”\n A 0 is obtained, use previous value if available\n”;
              if(ninpt<=nexmplr){
                        anrn[i].output = e[ninpt−1].v1[i];}
              else
                        {anrn[i].output = pp[0].v1[i];}
              outs1[i] = anrn[i].output;}
              cout<<”\n input layer neuron
              “<<i<<” output is “
                  <<anrn[i].output<<”\n”;
              }
}
void network::asgnvect(int j1,int *b1,int *b2)
{
int  j2;
for(j2=0;j2<j1;++j2){
Source File
166

b2[j2] = b1[j2];}
}
void network::prwts()
{
int i3,i4;
cout<<”\n  weights—  input layer to output layer: \n\n”;
for(i3=0;i3<anmbr;++i3){
        for(i4=0;i4<bnmbr;++i4){
                cout<<anrn[i3].outwt[i4]<<”                  “;}
        cout<<”\n”; }
cout<<”\n”;
cout<<”\nweights—  output layer to input layer: \n\n”;
for(i3=0;i3<bnmbr;++i3){
        for(i4=0;i4<anmbr;++i4){
                cout<<bnrn[i3].outwt[i4]<<”                  “;}
        cout<<”\n”;  }
cout<<”\n”;
}
void network::iterate()
{
int i1;
for(i1=0;i1<nexmplr;++i1){
        findassc(e[i1].v1);
        }
}
void network::findassc(int *b)
{
int j;
flag = 0;
        asgninpt(b);
ninpt ++;
cout<<”\nInput vector is:\n” ;
for(j=0;j<6;++j){
       cout<<b[j]<<” “;}
cout<<”\n”;
pp[0].getpotlpair(anmbr,bnmbr);
asgnvect(anmbr,outs1,pp[0].v1);
comput1();
if(flag>=0){
           asgnvect(bnmbr,outs2,pp[0].v2);
           cout<<”\n”;
           pp[0].prpotlpair();
Source File
167


---

## Chapter 8: BAM: Bidirectional Associative Memory
*(Pages 220-255)*

C++ Neural Networks and Fuzzy Logic
by Valluru B. Rao
MTBooks, IDG Books Worldwide, Inc.
ISBN: 1558515526   Pub Date: 06/01/95

Chapter 11
The Kohonen Self−Organizing Map
Introduction
This chapter discusses one type of unsupervised competitive learning, the Kohonen feature map, or
self−organizing map (SOM). As you recall, in unsupervised learning there are no expected outputs presented
to a neural network, as in a supervised training algorithm such as backpropagation. Instead, a network, by its
self−organizing properties, is able to infer relationships and learn more as more inputs are presented to it. One
advantage to this scheme is that you can expect the system to change with changing conditions and inputs.
The system constantly learns. The Kohonen SOM is a neural network system developed by Teuvo Kohonen
of Helsinki University of Technology and is often used to classify inputs into different categories.
Applications for feature maps can be traced to many areas, including speech recognition and robot motor
control.
Competitive Learning
A Kohonen feature map may be used by itself or as a layer of another neural network. A Kohonen layer is
composed of neurons that compete with each other. Like in Adaptive Resonance Theory, the Kohonen SOM
is another case of using a winner−take−all strategy. Inputs are fed into each of the neurons in the Kohonen
layer (from the input layer). Each neuron determines its output according to a weighted sum formula:
Output = £ wij xi
The weights and the inputs are usually normalized, which means that the magnitude of the weight and input
vectors are set equal to one. The neuron with the largest output is the winner. This neuron has a final output of
1. All other neurons in the layer have an output of zero. Differing input patterns end up firing different winner
neurons. Similar or identical input patterns classify to the same output neuron. You get like inputs clustered
together. In Chapter 12, you will see the use of a Kohonen network in pattern classification.
Normalization of a Vector
Consider a vector, A = ax + by + cz. The normalized vector A’ is obtained by dividing each component of A
by the square root of the sum of squares of all the components. In other words each component is multiplied
by 1/ [radic](a2 + b2 + c2). Both the weight vector and the input vector are normalized during the operation of
the Kohonen feature map. The reason for this is the training law uses subtraction of the weight vector from the
input vector. Using normalization of the values in the subtraction reduces both vectors to a unit−less status,
and hence, makes the subtraction of like quantities possible. You will learn more about the training law
shortly.
Chapter 11 The Kohonen Self−Organizing Map
220

Lateral Inhibition
Lateral inhibition is a process that takes place in some biological neural networks. Lateral connections of
neurons in a given layer are formed, and squash distant neighbors. The strength of connections is inversely
related to distance. The positive, supportive connections are termed as excitatory while the negative,
squashing connections are termed inhibitory.
A biological example of lateral inhibition occurs in the human vision system.
The Mexican Hat Function
Figure 11.1 shows a function, called the mexican hat function, which shows the relationship between the
connection strength and the distance from the winning neuron. The effect of this function is to set up a
competitive environment for learning. Only winning neurons and their neighbors participate in learning for a
given input pattern.
Figure 11.1  The mexican hat function showing lateral inhibition.
Training Law for the Kohonen Map
The training law for the Kohonen feature map is straightforward. The change in weight vector for a given
output neuron is a gain constant, alpha, multiplied by the difference between the input vector and the old
weight vector:
Wnew = Wold + alpha * (Input −Wold)
Both the old weight vector and the input vector are normalized to unit length. Alpha is a gain constant
between 0 and 1.
Significance of the Training Law
Let us consider the case of a two−dimensional input vector. If you look at a unit circle, as shown in Figure
11.2, the effect of the training law is to try to align the weight vector and the input vector. Each pattern
attempts to nudge the weight vector closer by a fraction determined by alpha. For three dimensions the surface
becomes a unit sphere instead of a circle. For higher dimensions you term the surface a hypersphere. It is not
necessarily ideal to have perfect alignment of the input and weight vectors. You use neural networks for their
ability to recognize patterns, but also to generalize input data sets. By aligning all input vectors to the
corresponding winner weight vectors, you are essentially memorizing the input data set classes. It may be
more desirable to come close, so that noisy or incomplete inputs may still trigger the correct classification.
Figure 11.2  The training law for the Kohonen map as shown on a unit circle.
The Mexican Hat Function
221


---

## Chapter 9: FAM: Fuzzy Associative Memory
*(Pages 255-280)*

C++ Neural Networks and Fuzzy Logic
by Valluru B. Rao
MTBooks, IDG Books Worldwide, Inc.
ISBN: 1558515526   Pub Date: 06/01/95

Generalization versus Memorization
As mentioned in Chapter 11, you actually don’t desire the exact replication of the input pattern for the weight
vector. This would amount to memorizing of the input patterns with no capacity for generalization.
For example, a typical use of this alphabet classifier system would be to use it to process noisy data, like
handwritten characters. In such a case, you would need a great deal of latitude in scoping a class for a letter A.
Adding Characters
The next step of the program is to add characters and see what categories they end up in. There are many
alphabetic characters that look alike, such as H and B for example. You can expect the Kohonen classifier to
group these like characters into the same class.
We now modify the input.dat file to add the characters H, B, and I. The new input.dat file is shown as follows.
0 0 1 0 0   0 1 0 1 0  1 0 0 0 1  1 0 0 0 1  1 1 1 1 1  1 0 0 0 1
 1 0 0 0 1
1 0 0 0 1   0 1 0 1 0  0 0 1 0 0  0 0 1 0 0  0 0 1 0 0  0 1 0 1 0
 1 0 0 0 1
1 0 0 0 1   1 0 0 0 1  1 0 0 0 1  1 1 1 1 1  1 0 0 0 1  1 0 0 0 1
 1 0 0 0 1
1 1 1 1 1   1 0 0 0 1  1 0 0 0 1  1 1 1 1 1  1 0 0 0 1  1 0 0 0 1
 1 1 1 1 1
0 0 1 0 0   0 0 1 0 0  0 0 1 0 0  0 0 1 0 0  0 0 1 0 0  0 0 1 0 0
 0 0 1 0 0
The output using this input file is shown as follows.
—————————————————————————−
       done
——>average dist per cycle = 0.732607 <—−
——>dist last cycle = 0.00360096 <—−
−>dist last cycle per pattern= 0.000720192 <—−
——————>total cycles = 37 <—−
——————>total patterns = 185 <—−
—————————————————————————−
The file kohonen.dat with the output values is now shown as follows.
cycle   pattern    win index   neigh_size    avg_dist_per_pattern
—————————————————————————————————————————————————————————————————
0       0          69          5             100.000000
0       1          93          5             100.000000
0       2          18          5             100.000000
0       3          18          5             100.000000
0       4          78          5             100.000000
Generalization versus Memorization
255

1       5          69          5             0.806743
1       6          93          5             0.806743
1       7          18          5             0.806743
1       8          18          5             0.806743
1       9          78          5             0.806743
2       10         69          5             0.669678
2       11         93          5             0.669678
2       12         18          5             0.669678
2       13         18          5             0.669678
2       14         78          5             0.669678
3       15         69          5             0.469631
3       16         93          5             0.469631
3       17         18          5             0.469631
3       18         18          5             0.469631
3       19         78          5             0.469631
4       20         69          5             0.354791
4       21         93          5             0.354791
4       22         18          5             0.354791
4       23         18          5             0.354791
4       24         78          5             0.354791
5       25         69          5             0.282990
5       26         93          5             0.282990
5       27         18          5             0.282990
...
35      179        78          5             0.001470
36      180        69          5             0.001029
36      181        93          5             0.001029
36      182        13          5             0.001029
36      183        19          5             0.001029
36      184        78          5             0.001029
Again, the network does not find a problem in classifying these vectors.
Until cycle 21, both the H and the B were classified as output neuron 18. The ability to
distinguish these vectors is largely due to the small tolerance we have assigned as a
termination criterion.


---

## Chapter 10: Adaptive Resonance Theory (ART)
*(Pages 280-320)*

cin >> number_of_layers;
cout << “ Enter in the layer sizes separated by spaces.\n”;
cout << “ For a network with 3 neurons in the input layer,\n”;
cout << “ 2 neurons in a hidden layer, and 4 neurons in the\n”;
cout << “ output layer, you would enter: 3 2 4 .\n”;
cout << “ You can have up to 3 hidden layers,for five maximum entries
:\n\n”;
for (i=0; i<number_of_layers; i++)
        {
        cin >> layer_size[i];
        }
// ———————————————————————————
// size of layers:
//    input_layer            layer_size[0]
//    output_layer           layer_size[number_of_layers−1]
//    middle_layers          layer_size[1]
//    optional: layer_size[number_of_layers−3]
//    optional: layer_size[number_of_layers−2]
//———————————————————————————−
}
void network::set_up_network()
{
int i,j,k;
//———————————————————————————−
// Construct the layers
//
//———————————————————————————−
layer_ptr[0] = new input_layer(0,layer_size[0]);
for (i=0;i<(number_of_layers−1);i++)
        {
        layer_ptr[i+1] =
        new middle_layer(layer_size[i],layer_size[i+1]);
        }
layer_ptr[number_of_layers−1] = new
output_layer(layer_size[number_of_layers−2], layer_size[number_of_
layers−1]);
for (i=0;i<(number_of_layers−1);i++)
        {
        if (layer_ptr[i] == 0)
               {
               cout << “insufficient memory\n”;
               cout << “use a smaller architecture\n”;
               exit(1);
               }
        }
//———————————————————————————−
// Connect the layers
//
//———————————————————————————−
// set inputs to previous layer outputs for all layers,
//             except the input layer
for (i=1; i< number_of_layers; i++)
Adding Noise During Training
280

layer_ptr[i]−>inputs = layer_ptr[i−1]−>outputs;
// for back_propagation, set output_errors to next layer
//             back_errors for all layers except the output
//             layer and input layer
for (i=1; i< number_of_layers −1; i++)
        ((output_layer *)layer_ptr[i])−>output_errors =
               ((output_layer *)layer_ptr[i+1])−>back_errors;
// define the IObuffer that caches data from
// the datafile
i=layer_ptr[0]−>num_outputs;// inputs
j=layer_ptr[number_of_layers−1]−>num_outputs; //outputs
k=MAX_VECTORS;
buffer=new
        float[(i+j)*k];
if (buffer==0)
        {
        cout << “insufficient memory for buffer\n”;
        exit(1);
        }
}
void network::randomize_weights()
{
int i;
for (i=1; i<number_of_layers; i++)
        ((output_layer *)layer_ptr[i])
                −>randomize_weights();
}
void network::update_weights(const float beta, const float alpha)
{
int i;
for (i=1; i<number_of_layers; i++)
        ((output_layer *)layer_ptr[i])
               −>update_weights(beta,alpha);
}
void network::update_momentum()
{
int i;
for (i=1; i<number_of_layers; i++)
        ((output_layer *)layer_ptr[i])
               −>update_momentum();
}
void network::write_weights(FILE * weights_file_ptr)
{
int i;
for (i=1; i<number_of_layers; i++)
        ((output_layer *)layer_ptr[i])
               −>write_weights(i,weights_file_ptr);
}
void network::read_weights(FILE * weights_file_ptr)
{
Adding Noise During Training
281


---

## Chapter 11: The Kohonen Self-Organizing Map
*(Pages 320-360)*

C++ Neural Networks and Fuzzy Logic
by Valluru B. Rao
MTBooks, IDG Books Worldwide, Inc.
ISBN: 1558515526   Pub Date: 06/01/95

On−Balance Volume
The on−balance volume (OBV) indicator was created to try to uncover accumulation and distribution patterns
of large player in the stock market. This is a cumulative sum of volume data, specified as follows:
If today’s close is greater than yesterday’s close
OBVt = OBVt−1 + 1
If today’s close is less than yesterday’s close
OBVt = OBVt−1 − 1
The absolute value of the index is not important; attention is given only to the direction and trend.
Accumulation−Distribution
This indicator does for price what OBV does for volume.
If today’s close is greater than yesterday’s close:
ADt = ADt−1 + (Closet − Lowt)
If today’s close is less than yesterday’s close
ADt = ADt−1 + (Hight − Closet)
Now let’s examine how these indicators look. Figure 14.7 shows a bar chart, which is a chart of price data
versus time, along with the following indicators:
•  Ten−unit moving average
•  Ten−unit exponential moving average
•  Momentum
•  MACD
•  Percent R
Figure 14.7  Five minute bar chart of the S&P 500 Sept 95 Futures contract with several technical indicators
displayed.
On−Balance Volume
320

The time period shown is 5 minute bars for the S&P 500 September 1995 Futures contract. The top of each
bar indicates the highest value (“high”) for that time interval, the bottom indicates the lowest value(“low”),
and the horizontal lines on the bar indicate the initial (“open”) and final (“close”) values for the time interval.
Figure 14.8 shows another bar chart for Intel Corporation stock for the period from December 1994 to July
1995, with each bar representing a day of activity. The following indicators are displayed also.
•  Rate of Change
•  Relative Strength
•  Stochastics
•  Accumulation−Distribution
Figure 14.8  Daily bar chart of Intel Corporation with several technical indicators displayed.
You have seen a few of the hundreds of technical indicators that have been invented to date. New indicators
are being created rapidly as the field of Technical Analysis gains popularity and following. There are also
pattern recognition studies, such as formations that resemble flags or pennants as well as more exotic types of
studies, like Elliot wave counts. You can refer to books on Technical Analysis (e.g., Murphy) for more
information about these and other studies.
Neural preprocessing with Technical Analysis tools as well as with traditional engineering analysis tools such
as Fourier series, Wavelets, and Fractals can be very useful in finding predictive patterns for forecasting.
What Others Have Reported
In this final section of the chapter, we outline some case studies documented in periodicals and books, to give
you an idea of the successes or failures to date with neural networks in financial forecasting. Keep in mind
that the very best (= most profitable) results are usually never reported (so as not to lose a competitive edge) !
Also, remember that the market inefficiencies exploited yesterday may no longer be the same to exploit today.
Can a Three−Year−Old Trade Commodities?
Well, Hillary Clinton can certainly trade commodities, but a three−year−old, too? In his paper, “Commodity
Trading with a Three Year Old,” J. E. Collard describes a neural network with the supposed intelligence of a
three−year−old. The application used a feedforward backpropagation network with a 37−30−1 architecture.
The network was trained to buy (“go long”) or sell (“go short”) in the live cattle commodity futures market.
The training set consisted of 789 facts for trading days in 1988, 1989, 1990, and 1991. Each input vector
consisted of 18 fundamental indicators and six market technical variables (Open, High, Low, Close, Open
Interest, Volume). The network could be trained for the correct output on all but 11 of the 789 facts.
The fully trained network was used on 178 subsequent trading days in 1991. The cumulative profit increased
from $0 to $1547.50 over this period by trading one live cattle contract. The largest loss in a trade was
$601.74 and the largest gain in a trade was $648.30.
What Others Have Reported
321


---

## Chapter 12: Application to Pattern Recognition
*(Pages 360-380)*

−70   −30  −70   −70  −70
−630  −70  −630  −30  −30
−870  −70  −70   −30  −70
−70   −30  −630  −70  −630
−30   −30  −70   −70  −70
−70  −70   −30  −70   −70
−30  −630  −70  −630  −30
−30  −70   −70  −70   −30
−70  −30   −30  −630  −70
−630 −30   −70  −70   −70
−70  −70   −70   −30   −70
−30  −30   −630  −70   −630
−30  −70   −70   −70   −70
−30  −630  −30   −30   −630
−70  −870  −70   −630  −30
−70   −70  −70   −70   −30
−630  −30  −30   −630  −70
−870  −70  −630  −30   −30
−630  −30  −70   −70   −70
−70   −70  −630  −70   −630
−70  −630  −30  −30   −630
−30  −70   −70  −70   −70
−70  −630  −70  −630  −30
−30  −70   −30  −70   −70
−70  −750  −30  −630  −70
−630  −70  −630  −30  −30
−70   −30  −70   −70  −70
−750  −30  −630  −70  −630
−30   −70  −70   −30  −70
−70   −30  −30   −30  −630
−30  −630  −70   −630  −30
−70  −70   −30   −70   −70
−30  −30   −30   −630  −70
−630 −70   −70   −70   −30
−70  −30   −630  −30   −30
−30  −30   −630  −70   −630
−70  −70   −70   −30   −70
−30  −630  −30   −30   −630
−70  −70   −70   −70   −70
−30  −750  −70   −870  −30
−630  −30  −30   −630  −70
−70   −70  −70   −70   −30
−750  −70  −870  −30   −30
−870  −70  −750  −30   −30
−750  −30  −870  −70   −870
−70  −870  −30  −30   −870
−70  −750  −30  −30   −750
−30  −870  −70  −870  −30
−30  −750  −70  −750  −30
−30  −70   −30  −870  −70
−870  −70  −870  −30  −30
Output from Your C++ Program for the Traveling Salesperson Problem
360

−750  −70  −750  −30  −30
−70   −30  −870  −70  −870
−30   −30  −750  −70  −750
−30   −70  −30   −30  −870
−30   −870  −70  −870  −30
−30   −750  −70  −750  −30
−70   −30   −30  −870  −70
−870  −30   −30  −750  −70
−750  −70   −870 −30   −30
−30  −30   −870  −70   −870
−30  −30   −750  −70   −750
−70  −870  −30   −30   −870
−70  −750  −30   −30   −750
−70  −70   −70   −450  −30
−870  −30  −30   −870  −70
−750  −30  −30   −750  −70
−70   −70  −450  −30   −30
−450  −70  −570  −30   −30
−570  −70  −450  −70   −450
−70  −450  −30  −30   −450
−70  −570  −30  −30   −570
−70  −450  −70  −450  −30
−30  −570  −70  −570  −30
−30  −1470 −30  −450  −70
−450  −70  −450  −30  −30
−570  −70  −570  −30  −30
−1470 −30  −450  −70  −450
−30   −30  −570  −70  −570
−30   −30  −30   −30  −450
−30  −450  −70   −450  −30
−30  −570  −70   −570  −30
−30  −30   −30   −450  −70
−450 −30   −30   −570  −70
−570 −30   −450  −30   −30
−30  −30    −450  −70   −450
−30  −30    −570  −70   −570
−30  −450   −30   −30   −450
−70  −570   −30   −30   −570
−70  −1470  −70   −390  −30
−450   −30   −30    −450  −70
−570   −30   −30    −570  −70
−1470  −70   −390   −30   −30
−390   −70   −1110  −30   −30
−1110  −70   −390   −70   −390
−70  −390   −30  −30    −390
−70  −1110  −30  −30    −1110
−70  −390   −70  −390   −30
−30  −1110  −70  −1110  −30
−30  −990   −30  −390   −70
−390   −70  −390   −30  −30
−1110  −70  −1110  −30  −30
−990   −30  −390   −70  −390
Output from Your C++ Program for the Traveling Salesperson Problem
361

−30    −30  −1110  −70  −1110
−30    −30  −30    −30  −390
−30    −390   −70   −390   −30
−30    −1110  −70   −1110  −30
−30    −30    −30   −390   −70
−390   −30    −30   −1110  −70
−1110  −30    −390  −30    −30
−30  −30    −390   −70  −390
−30  −30    −1110  −70  −1110
−30  −390   −30    −30  −390
−70  −1110  −30    −30  −1110
−70  −990   −30    −30  −990
−390  −30  −30  −390   −70
−1110 −30  −30  −1110  −70
−990  −30  −30  −990   −70
−1950 −30  −30  −1950  −70
−70   −70  −70  −70    −30
initial activations
 the activations:
−290.894989  −311.190002  −218.365005  −309.344971  −467.774994
−366.299957  −421.254944  −232.399963  −489.249969  −467.399994
−504.375     −552.794983  −798.929871  −496.005005  −424.964935
−374.639984  −654.389832  −336.049988  −612.870056  −405.450012
−544.724976  −751.060059  −418.285034  −545.465027  −500.065063
the outputs
0.029577  0.023333  0.067838  0.023843  0.003636
0.012181  0.006337  0.057932  0.002812  0.003652
0.002346  0.001314  6.859939e−05  0.002594  0.006062
0.011034  0.000389  0.017419  0.000639  0.00765
0.001447  0.000122  0.006565  0.001434  0.002471
40 iterations completed
 the activations:
−117.115494  −140.58519   −85.636215   −158.240143  −275.021301
−229.135956  −341.123871  −288.208496  −536.142212  −596.154297
−297.832794  −379.722595  −593.842102  −440.377625  −442.091064
−209.226883  −447.291016  −283.609589  −519.441101  −430.469696
−338.93219   −543.509766  −386.950531  −538.633606  −574.604492
the outputs
0.196963  0.156168  0.263543  0.130235  0.035562
0.060107  0.016407  0.030516  0.001604  0.000781
0.027279  0.010388  0.000803  0.005044  0.004942
0.07511   0.004644  0.032192  0.001959  0.005677
0.016837  0.001468  0.009533  0.001557  0.001012
tourcity 0 tour order 2
tourcity 1 tour order 0
tourcity 2 tour order 1
tourcity 3 tour order 4
tourcity 4 tour order 3
Output from Your C++ Program for the Traveling Salesperson Problem
362


---

## Chapter 13: Backpropagation II
*(Pages 380-400)*

C++ Neural Networks and Fuzzy Logic
by Valluru B. Rao
MTBooks, IDG Books Worldwide, Inc.
ISBN: 1558515526   Pub Date: 06/01/95

Databases and Queries
Imagine that you are interested in the travel business. You may be trying to design special tours in different
countries with your own team of tour guides, etc. , and you want to identify suitable persons for these
positions. Initially, let us say, you are interested in their own experiences in traveling, and the knowledge they
possess, in terms of geography, customs, language, and special occasions, etc. The information you want to
keep in your database may be something like, who the person is, the person’s citizenship, to where the person
traveled, when such travel occurred, the length of stay at that destination, the person’s languages, the
languages the person understands, the number of trips the person made to each place of travel, etc. Let us use
some abbreviations:
cov—country visited
lov—length of visit (days)
nov—number of visits including previous visits
ctz—citizenship
yov—year of visit
lps—language (other than mother tongue) with proficiency to speak
lpu—language with only proficiency to understand
hs—history was studied (1—yes, 0—no)
Typical entries may appear as noted in Table 16.1.
Table 16.1 Example Database
Nameagectzcovlovnovyovlpslpuhs
John Smith35U.S.India411994Hindi1
John Smith35U.S.Italy721991Italian1
John Smith35U.S.Japan3119930
When a query is made to list persons that visited India or Japan after 1992 for 3 or more days, John Smith’s
two entries will be included. The conditions stated for this query are straightforward, with lov [ge] 3 and yov
> 1992 and (cov = India or cov = Japan).
Relations in Databases
A relation from this database may be the set of quintuples, (name, age, cov, lov, yov). Another may be the set
of triples, (name, ctz, lps). The quintuple (John Smith, 35, India, 4, 1994) belongs to the former relation, and
the triple (John Smith, U.S., Italian) belongs to the latter. You can define other relations, as well.
Databases and Queries
380

Fuzzy Scenarios
Now the query part may be made fuzzy by asking to list young persons who recently visited Japan or India for
a few days. John Smith’s entries may or may not be included this time since it is not clear if John Smith is
considered young, or whether 1993 is considered recent, or if 3 days would qualify as a few days for the
query. This modification of the query illustrates one of three scenarios in which fuzziness can be introduced
into databases and their use.
This is the case where the database and relations are standard, but the queries may be fuzzy. The other cases
are: one where the database is fuzzy, but the queries are standard with no ambiguity; and one where you have
both a fuzzy database and some fuzzy queries.
Fuzzy Sets Revisited
We will illustrate the concept of fuzziness in the case where the database and the queries have fuzziness in
them. Our discussion is guided by the reference Terano, Asai, and Sugeno. First, let us review and recast the
concept of a fuzzy set in a slightly different notation.
If a, b, c, and d are in the set A with 0.9, 0.4, 0.5, 0, respectively, as degrees of membership, and in B with
0.9, 0.6, 0.3, 0.8, respectively, we give these fuzzy sets A and B as A = { 0.9/a, 0.4/b, 0.5/c} and B = {0.9/a,
0.6/b, 0.3/c, 0.8/d}. Now A[cup]B = {0.9/a, 0.6/b, 0.5/c, 0.8/d} since you take the larger of the degrees of
membership in A and B for each element. Also, A[cap]B = {0.9/a, 0.4/b, 0.3/c} since you now take the
smaller of the degrees of membership in A and B for each element. Since d has 0 as degree of membership in
A (it is therefore not listed in A), it is not listed in A[cap]B.
Let us impart fuzzy values (FV) to each of the attributes, age, lov, nov, yov, and hs by defining the sets in
Table 16.2.
Table 16.2 Fuzzy Values for Example Sets
Fuzzy ValueSet
FV(age){ very young, young, somewhat old, old }
FV(nov){ never, rarely, quite a few, often, very often }
FV(lov){ barely few days, few days, quite a few days, many days }
FV(yov){distant past, recent past, recent }
FV(hs){ barely, adequately, quite a bit, extensively }
The attributes of name, citizenship, country of visit are clearly not candidates for having fuzzy values. The
attributes of lps, and lpu, which stand for language in which speaking proficiency and language in which
understanding ability exist, can be coupled into another attribute called flp (foreign language proficiency) with
fuzzy values. We could have introduced in the original list an attribute called lpr ( language with proficiency
to read) along with lps and lpu. As you can see, these three can be taken together into the fuzzy−valued
attribute of foreign language proficiency. We give below the fuzzy values of flp.
   FV(flp) = {not proficient, barely proficient, adequate,
             proficient, very proficient }
Note that each fuzzy value of each attribute gives rise to a fuzzy set, which depends on the elements you
consider for the set and their degrees of membership.
Fuzzy Scenarios
381


---

## Chapter 14: Application to Financial Forecasting
*(Pages 400-425)*

C++ Neural Networks and Fuzzy Logic
by Valluru B. Rao
MTBooks, IDG Books Worldwide, Inc.
ISBN: 1558515526   Pub Date: 06/01/95

Section II: Fuzzy Control
This section discusses the fuzzy logic controller (FLC), its application and design. Fuzzy control is used in a
variety of machines and processes today, with widespread application especially in Japan. A few of the
applications in use today are in the list in Table 16.10, adapted from Yan, et al.
Table 16.10 Applications of Fuzzy Logic Controllers (FLCs) and Functions Performed
ApplicationFLC function(s)
Video camcorderDetermine best focusing and lighting when there is movement in the picture
Washing machineAdjust washing cycle by judging the dirt, size of the load, and type of fabric
TelevisionAdjust brightness, color, and contrast of picture to please viewers
Motor controlImprove the accuracy and range of motion control under unexpected conditions
Subway trainIncrease the stable drive and enhance the stop accuracy by evaluating the passenger traffic
conditions. Provide a smooth start and smooth stop.
Vacuum cleanerAdjust the vacuum cleaner motor power by judging the amount of dust and dirt and the floor
characteristics
Hot water heaterAdjust the heating element power according to the temperature and the quantity of water
being used
Helicopter controlDetermine the best operation actions by judging human instructions and the flying
conditions including wind speed and direction
Designing a Fuzzy Logic Controller
A fuzzy logic controller diagram was shown in Chapter 3. Let us redraw it now and discuss a design example.
Refer to Figure 16.2. For the purpose of discussion, let us assume that this FLC controls a hot water heater.
The hot water heater has a knob, HeatKnob(0−10) on it to control the heating element power, the higher the
value, the hotter it gets, with a value of 0 indicating the heating element is turned off. There are two sensors in
the hot water heater, one to tell you the temperature of the water (TempSense), which varies from 0 to 125° C,
and the other to tell you the level of the water in the tank (LevelSense), which varies from 0 = empty to 10 =
full. Assume that there is an automatic flow control that determines how much cold water (at temperature 10°
C) flows into the tank from the main water supply; whenever the level of the water gets below 40, the flow
control turns on, and turns off when the level of the water gets above 95.
Figure 16.2  Fuzzy control of a water heater.
The design objective can be stated as:
Keep the water temperature as close to 80° C as possible, in spite of changes in the water flowing out of
the tank, and cold water flowing into the tank.
Section II: Fuzzy Control
400

Step One: Defining Inputs and Outputs for the FLC
The range of values that inputs and outputs may take is called the universe of discourse. We need to define the
universe of discourse for all of the inputs and outputs of the FLC, which are all crisp values. Table 16.11
shows the ranges:
Table 16.11 Universe of Discourse for Inputs and Outputs for FLC
NameInput/OutputMinimum valueMaximum value
LevelSenseI010
HeatKnobO010
TempSenseI0125
Step Two: Fuzzify the Inputs
The inputs to the FLC are the LevelSense and the TempSense. We can use triangular membership functions to
fuzzify the inputs, just as we did in Chapter 3, when we constructed the fuzzifier program. There are some
general guidelines you can keep in mind when you determine the range of the fuzzy variables as related to the
crisp inputs (adapted from Yan, et al.):
1.  Symmetrically distribute the fuzzified values across the universe of discourse.
2.  Use an odd number of fuzzy sets for each variable so that some set is assured to be in the middle.
The use of 5 to 7 sets is fairly typical.
3.  Overlap adjacent sets (by 15% to 25% typically) .
Both the input variables LevelSense and TempSense are restricted to positive values. We use the following
fuzzy sets to describe them:
XSmall, Small, Medium, Large, XLarge
In Table 16.12 and Figure 16.3, we show the assignment of ranges and triangular fuzzy membership functions
for LevelSense. Similarly, we assign ranges and triangular fuzzy membership functions for TempSense in
Table 16.13 and Figure 16.4. The optimization of these assignments is often done through trial and error for
achieving optimum performance of the FLC.
Table 16.12 Fuzzy Variable Ranges for LevelSense
Crisp Input RangeFuzzy Variable
0–2XSmall
1.5–4Small
3–7Medium
6–8.5Large
7.5–10XLarge
Figure 16.3  Fuzzy membership functions for LevelSense.
Table 16.13 Fuzzy Variable Ranges for TempSense
Step One: Defining Inputs and Outputs for the FLC
401


---

## Chapter 15: Application to Nonlinear Optimization
*(Pages 425-440)*

Fit vector
A vector of values of degree of membership of elements of a fuzzy set.
Fully connected network
A neural network in which every neuron has connections to all other neurons.
Fuzzy
As related to a variable, the opposite of crisp. A fuzzy quantity represents a range of value as opposed
to a single numeric value, e.g., “hot” vs. 89.4°.
Fuzziness
Different concepts having an overlap to some extent. For example, descriptions of fair and cool
temperatures may have an overlap of a small interval of temperatures.
Fuzzy Associative Memory
A neural network model to make association between fuzzy sets.
Fuzzy equivalence relation
A fuzzy relation (relationship between fuzzy variables) that is reflexive, symmetric, and transitive.
Fuzzy partial order
A fuzzy relation (relationship between fuzzy variables) that is reflexive, antisymmetric, and transitive.
G
Gain
Sometimes a numerical factor to enhance the activation. Sometimes a connection for the same
purpose.
Generalized Delta rule
A rule used in training of networks such as backpropagation training where hidden layer weights are
modified with backpropagated error.
Global minimum
A point where the value of a function is no greater than the value at any other point in the domain of
the function.
H
Hamming distance
The number of places in which two binary vectors differ from each other.
Hebbian learning
A learning algorithm in which Hebb’s rule is used. The change in connection weight between two
neurons is taken as a constant times the product of their outputs.
Heteroassociative
Making an association between two distinct patterns or objects.
Hidden layer
An array of neurons positioned in between the input and output layers.
Hopfield network
A single layer, fully connected, autoassociative neural network.
I
Inhibition
The attempt by one neuron to diminish the chances of firing by another neuron.
Input layer
An array of neurons to which an external input or signal is presented.
Instar
Glossary
425

A neuron that has no connections going from it to other neurons.
L
Lateral connection
A connection between two neurons that belong to the same layer.
Layer
An array of neurons positioned similarly in a network for its operation.
Learning
The process of finding an appropriate set of connection weights to achieve the goal of the network
operation.
Linearly separable
Two subsets of a linear set having a linear barrier (hyperplane) between the two of them.
LMS rule
Least mean squared error rule, with the aim of minimizing the average of the squared error. Same as
the Delta rule.
Local minimum
A point where the value of the function is no greater than the value at any other point in its
neighborhood.
Long−term memory (LTM)
Encoded information that is retained for an extended period.
Lyapunov function
A function that is bounded below and represents the state of a system that decreases with every
change in the state of the system.
M
Madaline
A neural network in which the input layer has units that are Adalines. It is a multiple−Adaline.
Mapping
A correspondence between elements of two sets.
N
Neural network
A collection of processing elements arranged in layers, and a collection of connection edges between
pairs of neurons. Input is received at one layer, and output is produced at the same or at a different
layer.
Noise
Distortion of an input.
Nonlinear optimization
Finding the best solution for a problem that has a nonlinear function in its objective or in a constraint.
O
On center off surround
Assignment of excitatory weights to connections to nearby neurons and inhibitory weights to
connections to distant neurons.
Orthogonal vectors
Vectors whose dot product is 0.
Glossary
426


---

## Chapter 16: Applications of Fuzzy Logic
*(Pages 440-450)*

input, 98
binary input, 98
bipolar input, 98
layer, 2, 10
nature of , 73
number of , 74
patterns, 51, 65
signals, 65
space, 124
vector, 53, 71, 272, 112
input/output, 71
inqreset function, 251
instar, 93
interactions, 94
interconnections, 7
interest rate, 387
internal activation , 3
intersection, 32, 33
inverse mapping, 62, 182
Investor’s Business Daily, 388
iostream, 54, 71
istream, 58
iterative process, 78
J
Jagota, 514
January effect, 380
Jurik, 381, 384, 392
K
Karhunen−Loev transform, 384
Katz, 377
Kimoto, 408
kohonen.dat file, 275, 298, 300, 317
Kohonen, 19, 116, 117, 245, 271, 303, 456
Kohonen feature map, 16, 271, 273, 303, 305, 323
conscience factor, 302
neighborhood size, 280, 299, 300
training law, 273
Kohonen layer, 9, 19, 82, 92, 106, 298, 302, 322
class, 276
Kohonen network, 275, 276, 280, 300, 303, 322
applications of, 302,
Kohonen output layer, 275
Kohonen Self−Organizing Map, 115, 456, 471, 472
Kosaka, 409,
Kosko, 49, 50, 104, 215, 242, 506
Kostenius, 408, 409
Index
440

Kronecker delta function 428, 524
L
lambda, 136, 433
late binding, 24
lateral, 93
lateral competition, 303
laterally connected, 65
lateral connections, 93, 97, 107, 272, 276
lateral inhibition, 272, 276
layer, 2, 81
C layer, 106
comparison, 244
complex layer, 106
F1, 244
F2, 244
Grossberg layer, 82, 92, 302
hidden layer, 75, 81, 86, 89
input layer, 2, 3, 82
Kohonen layer, 82, 92, 302, 322
middle layer, 329, 372
output layer, 2, 82
recognition, 244
S layer, 106
simple layer, 106
layout, 52, 86, 124
ART1, 244
BAM , 180
Brain−State−in−a−Box, 105
FAM, 219
Hopfield network, 11
for TSP, 427
LVQ, 117
Madaline model, 103
LBS Capital Management, 377
learning, 4, 74, 98, 109, 110, 117, 118
algorithm, 61, 79, 102, 118
cycle, 103
Hebbian, 105, 110
one−shot, 117
probabilistic, 113
rate(parameter), 111, 112, 123, 125, 127, 136, 175
supervised learning, 5, 110, 112, 117, 121
time, 120
unsupervised
competitive learning, 271
learning, 5, 110, 117, 121
Learning Vector Quantizer, 115−117, 302
least mean squared error, 111, 119, 123, 419
Index
441

rule, 111
Le Cun, 375
Lee, 512
Levenberg−Marquardt optimization, 373
Lewis, 377
Lin, 512
linear function, 99, 102
linear possibility regression model, 493, 496, 509
linear programming, 417
integer, 417
linearly separable, 83− 85
LMS see least mean squared error rule
local minimum, 113, 177, 325
logic
boolean logic, 50
fuzzy logic, 31, 34, 50, 473
logical operations, 31
AND, 64
logistic function, 86, 100
Long−term memory, 6, 77− 79, 118, 472
traces of, 243
look−up
memory, 5
table, 106
LTM see Long−term memory
LVQ see Learning Vector Quantizer
Lyapunov Function, 118, 119
M
MACD see moving average convergence divergence
Madaline, 102, 103
main diagonal, 63, 480
malignant melanoma, 514
malloc, 24
Mandelman, 378
MAPE see mean absolute percentage error
mapping, 123, 180
binary to bipolar, 62, 63
inverse, 62, 182
nonlinear, 109
real to binary, 180
mapping surface, 109
Markowitz, 470
Marquez, 406
Mason, 516
matrix, 97, 521
addition, 521
correlation matrix, 9
fuzzy, 217
Index
442


---

## Chapter 17: Further Applications
*(Pages 450-454)*

signal filtering, 102
signals
analog, 98
similarity, 486
class, 481, 509
level, 486
relation, 481
simple cells, 106
simple moving average, 399
simulated annealing, 113, 114
simulator, 372, 396
controls, 173
mode, 138
Skapura, 246, 248
Slater, 515
S layer, 106
SMA see simple moving average
SOM see Self−Organizing Map
sonar target recognition, 374
spatial
pattern, 99, 214
temporal pattern, 105,
speech
recognition, 303,
synthesizer, 374,
spike, 380
squared error, 103
squashing
function, 384, 458, 459
stable, 79, 107
stability 78, 79, 118
and plasticity, 77
stability−plasticity dilemma, 79, 107,, 269
STM see Short Term Memory
Standard and Poor’s 500 Index, 377, 378
forecasting, 386
standard I/O routines, 519
state energy, 118
state machine, 48
static binding, 139
Steele, 514
steepest descent, 112, 113, 177, 373
step function, 99, 101
stochastics, 402, 404
Stoecker, 514
Stonham, 516
string
binary, 62
bipolar, 62
structure, 7, 7
subsample, 322
Index
450

subset, 221
subsystem
attentional, 107, 243
orienting, 107, 243
Sudjianto, 516
summand, 422
summation symbol, 422
supervised , 109
learning, 5, 110, 112, 115, 117, 121
training 94, 110, 115, 121, 125
Sweeney , 516
symbolic approach, 6
T
TSR see Terminate and Stay Resident
Tabu , 471
active, 471
neural network, 471
search, 471, 472
Tank, 422, 427, 429
target 378, 395
outputs, 110, 115
patterns, 105
scaled, 395
tau, 433
technical analysis, 399
temperature, 118
Temporal Associative Memory, 92
Terano, 496
Terminate and Stay Resident programs, 519
terminating value, 298
termination criterion, 322
test.dat file, 327, 328
test mode, 135, 137, 138, 164, 173, 327, 396
Thirty−year Treasury Bond Rate, 387
Three−month Treasury Bill Rate, 387
threshold
function, 2, 3, 12, 17, 19, 52, 95, 99, 101, 125, 183
value, 16, 52, 66, 77, 86, 87, 90, 101, 128, 456
thresholding, 87, 185
function, 133, 177, 182, 184, 214
Thro, 508
Tic−Tac−Toe, 76, 79
time lag, 380
time series forecasting, 406, 410
time shifting, 395
timeframe, 378
tolerance, 119, 125, 173, 245, 318, 322, 328, 329, 372
level, 78, 123
Index
451

value, 119
top−down
connection weight , 248
connections, 107, 244
top−down inputs, 247
topology, 7
Topology Preserving Maps, 116
tour, 420
traces, 243
of STM, 243
of LTM, 243
trading
commodities, 405,
system, 378
dual confirmation, 408
training, 4, 74, 75, 98, 109, 110, 119, 181, 396
fast, 107
law, 272, 273, 274, 330, 333
mode, 135, 137, 138, 164, 173, 396
supervised, 94, 110, 115
slow, 107
time, 329
unsupervised, 107, 110
transpose
of a matrix, 11, 179, 181, 183
of a vector, 11, 63, 97, 181
traveling salesperson(salesman) problem, 118, 119, 419
hand calculation, 423
Hopfield network solution−Hopfield, 427
Hopfield network solution−Anzai, 456
Kohonen network solution, 456
triple, 217
truth value, 31
tsneuron class, 430
TS see Tabu search
TSP see traveling salesperson problem
turning point predictor, 409
turning points, 407
two−layer networks, 92
two−thirds rule, 107, 244, 245, 269
U
Umano, 486
Unemployment Rate, 387
undertrained network, 329
uniform distribution, 77
union, 32
Unipolar Binary Bi−directional Associative Memory, 212
unit
Index
452


---

