# coding=utf-8
#
#4792aab4-bcb8-11e7-a937-00505601122b
#e47d7ca8-23a9-11e8-9de3-00505601122b
source_1 = """#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name=\"images\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")

            hidden_layer = self.images
            for element in args.cnn.split(','):
                t_args = element.split('-')
                if (t_args[0] == 'C'):
                    filters = t_args[1]
                    kernel_size = int(t_args[2])
                    stride = int(t_args[3])
                    padding = t_args[4]
                    hidden_layer = tf.layers.conv2d(hidden_layer, filters, kernel_size, stride, padding,
                                                    activation=tf.nn.relu)
                elif (t_args[0] == 'M'):
                    kernel_size = int(t_args[1])
                    stride = int(t_args[2])
                    hidden_layer = tf.layers.max_pooling2d(hidden_layer, kernel_size, stride)
                elif (t_args[0] == 'F'):
                    hidden_layer = tf.layers.flatten(hidden_layer, name=\"flatten\")
                elif (t_args[0] == 'R'):
                    hidden_layer_size = t_args[1]
                    hidden_layer = tf.layers.dense(hidden_layer, hidden_layer_size, activation=tf.nn.relu)
                elif (t_args[0] == 'CB'):
                    filters = t_args[1]
                    kernel_size = int(t_args[2])
                    stride = int(t_args[3])
                    padding = t_args[4]
                    hidden_layer = tf.layers.conv2d(hidden_layer, filters, kernel_size, stride, padding,
                                                    activation=None, use_bias=False)
                    hidden_layer = tf.layers.batch_normalization(hidden_layer, training=self.is_training)
                    hidden_layer = tf.nn.relu(hidden_layer)

            output_layer = tf.layers.dense(hidden_layer, self.LABELS, activation=None, name=\"output_layer\")
            self.predictions = tf.argmax(output_layer, axis=1)
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope=\"loss\")



            # Training
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                global_step = tf.train.create_global_step()
                if args.learning_rate_final == None:
                    learning_rate = args.learning_rate
                else:
                    decay_rate = (args.learning_rate_final / args.learning_rate) ** (1 / (args.epochs - 1))
                    decay_steps = mnist.train.num_examples // args.batch_size
                    learning_rate = tf.train.exponential_decay(staircase=True, learning_rate=args.learning_rate,
                                                               global_step=global_step, decay_rate=decay_rate,
                                                               decay_steps=decay_steps)
                self.training = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step, name=\"training\")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", loss),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels):
        self.session.run([self.training, self.summaries[\"train\"]], {self.images: images, self.labels: labels, self.is_training:True})

    def evaluate(self, dataset, images, labels):
        accuracy,predictions, _ = self.session.run([self.accuracy,self.predictions, self.summaries[dataset]], {self.images: images, self.labels: labels,self.is_training:False})
        return accuracy,predictions

if __name__ == \"__main__\":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--batch_size\", default=100, type=int, help=\"Batch size.\")
    #parser.add_argument(\"--cnn\", default='CB-20-3-2-same,M-3-2,CB-30-3-2-same,M-3-2,F,R-1024,R-10', type=str, help=\"Description of the CNN architecture.\")
    #parser.add_argument(\"--cnn\", default='CB-10-3-2-same,M-3-2,F,R-100', type=str, help=\"Description of the CNN architecture.\")
    parser.add_argument(\"--cnn\", default='CB-64-7-1-valid,M-3-2-same,CB-128-3-1-same,F,R-1024', type=str, help=\"Description of the CNN architecture.\")

    parser.add_argument(\"--epochs\", default=10, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--threads\", default=1, type=int, help=\"Maximum number of threads to use.\")
    parser.add_argument(\"--learning_rate\", default=0.001, type=float, help=\"Initial learning rate.\")
    parser.add_argument(\"--learning_rate_final\", default=0.0005, type=float, help=\"Final learning rate.\")

    args = parser.parse_args()

    # Create logdir name
    args.logdir = \"logs/{}-{}-{}\".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
        \",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists(\"logs\"): os.mkdir(\"logs\") # TF 1.6 will do this by itself

    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets(\"mnist-gan\", reshape=False, seed=42,
                                            source_url=\"https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/mnist-gan/\")

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        accuracy = network.evaluate(\"dev\", mnist.validation.images, mnist.validation.labels)[0]
        print(\"{:.2f}\".format(100 * accuracy))

    # TODO: Compute test_labels, as numbers 0-9, corresponding to mnist.test.images
    test_labels =network.evaluate(\"test\", mnist.test.images, mnist.test.labels)[1]
    with open(\"mnist_competition_test.txt\", \"w\") as test_file:
        for label in test_labels:
            print(label, file=test_file)
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;7%VCT3rAc0)oax6*GjyvYTC`s}Z?E4>dgkac{%n=7fv}Z9C2yLvSj;GvI6~#Qz^B=Kxj*Ia&F3MB2f{gwdr{4~tf$EI}4o<!;4NT=N>nuIPZA#~PPtQYS|;kV4XQM>rj7JADvh33btX_I{guwAo-+oyaR_cMw0}z@<~iunTtGpK)wV5Briauc7KKV-aM?$7G$x7P8HG|2oaAAA8>LX=oTE=QA6eIXHi(X1yF4$`w$u=YU`!PS>E6O4LeNGHU`R4in~$mYWxG)f9H;P3KJkldK}0j0P*;#3r}z`sY~3T-g*``BrKv^8OJ@+oxBYPn_bUXHXQVT;q219FG^jM5IOx)CO)EW*7OtsRnqUw42!Y=h{f)zVw4fmE#qQALDN9%|oB0yp`4ct1PkU-spOUCrE!J**%xQY)BWIW`f}yx~Lm{@Hs-*(LzYQ6LKw$+S7R0!7JwL_h;-*@yI4MmNqn%7j5~4nF0?76;^Z|$S!vJklNgbJ#}9zAxT`ktWgx-Lq3~wf~9H7dtk=%8b%nwO@zzNNG%x#p|cxmQASTif^%TCGCUD36#Ad@NmVpu^|W(oXJ5}tUkOiHVJKQQ(((+lUu#!SCO%}FjwiNyOkWeW(Lk{&LI#tQ<24f@*yjD`ePw7XSM8?bz({nuy+ujsYGkj^OW#1*=R7cV#ht(4{pk?u|5oF;cwoD4i@)1;>&C(bb8y&`sXcUX{)|a&Vp089Ei@C46k_#qS2g`q|J%V+Gz**ldkjOXgGqHT_ZxzuMNw8#yD5Tz;+gJcXNe%Xss2@bJVejjl<}IXm3vUa$B+H!)^vWj^q^*7%?^V)XC2<Q1hx681+5mHd>{TNbp-oTqjO!V5}OvS>p`WhJo&`1@^xRKQV>~=qWEO2IqLhod2%38`-7t82hR_oHqf`jGQcDMG;Ma2c}Pc<weddhl~1}#I&uci%_HQxU}dABhTL-MMSt}`J&v-|8J-84yH8B9h}q&WWGDz>Gs6!lNY1#b9?UrVOT%;!?Jos{l+J)FS~~7}GJT~RJD;$Es@U|_gzMyr^0zmragnAUIwrnN-?}x_N$La<uH+RJ;3Fq`6r!sEfvPB+HD)q*yFtk){_?5WScu|rpn{on8`eApwdhJZ3{;=Qund4V;yS4~O*MJR=K+QfI>%On2)*B5@4Jf_dxkgafmhHmab^f#nuG|8Y!v*WuvomQ2l^;J2-zhLdm`CetyMXZaS{j;PI6IEFjfyrjF?G!dm0Tx98mPCNtR>KY$)d{Sb6AgRBzXhP27)NOPQ~6WWu(ig&Bug`>f*I_*fmJZ;f{>&5-nd2+KR~HTeH{JjZ7tqXd`X4^PVxl>%(cjF9bnh-%o;Ig_naDlh&VsPru;p)l->H3bBUytp;cFTwp-<%$@J_bFFJ*<o_{B@on@wQ?_wA5aLuwfI6OaTZ-?=`o)NZyL|sp?0nA0-YVaX)}coiA~q*+a8Atc!LwQcjecLQVS9=<*tTTq91F`?29wfM!6h{5gvu6F&z`?+Evrha~wx59~gh{eR($<%2HIRpJUN?w9|qmy$tG>u<bg3Dyi-Z286*3;&RI=-*DLXzVIi&9v~cJ1Bw%iZr=u0pWj9G#Y;`?Ae|Ve2`qf>kC!hIVGsPLaLpqXwu6eVgXDpW2lY!=>OM4<U>**{i(?EPIXzQ8B<lO!9rC4%ecm_p@4LKm@r$qeYR#vco7}?XfJi!i?@SDgTmJGzcfTG}rb}e|Mvit9tP_1uuSj7plv6FCKH1jKRevhX+g{_usmm-aG=(ksFOkp({44zRiDcfGR4(D~btcN+Sm81c1fY#B3myfI-;98nnx>3w$d83>O$%1I2&L9aKd}t@qGvuRB7x@u7lKv)+p!_Le?;rlwg(mVhtJ(cDO(Vc1>h+9n1-c-B^5rLounZ}{g0&;*G2{BWtYyds24_Dn%GSHji`{XeGE>z;)8qjx|;Jigj)2W`ZvNH=w#))_KF}Pn;vnC&pSg>3CZX~gFyGC(J(umpC$t2aZs{(2;UiKg7JTNFHUvg%5WrApQ1Zpe*LsVgyW)H#==MEb@`BXMM*-thCfIo?XD*c;PBlU2TL$Xfc%NS>-e&t&mIfYpC&iYkpOy{G(j}v=|!jtfXUgxnIJTMh~YVc?3wm1tRaT01%QBN{INYCX&ye8-<OKR_S?bBlGIq9THAY;iAi&QDh_ohe-*3jq}x9<{9U`tbcUV%lIFHWcmB6rl%E;WW|sn9!45!O?=?8-pVf+m!c*+&-sv*BSu#rjLG0eH&E0a9t;yX#K-s95u$RCk(4bgq7uFkM{F<K!02KXb>X~0+;lhvshJmuq1sA8G7FdLtoaa6)pY5l8EdQ~(!`@>`BrrM6MeIWh7?_i|HSAvt0*wb6WQ(9d0fFa0O2zakIo&Edi1R5yr^K($za}-#BYjLWB{Wd4b<rHJdCjC$WRK0b?85#)f#*XuGz(`-kl3vp?2RIE^IH^fdh}#P%yLH*u>j$Tlzt5?o&5V}t^T+HmvHd&V*p&xohIpJ*_?1v6Rh&cSohAg(pEK&xDXwu#4K9r)~OkRm-D^632v;jc_QZ8kR)Fz!fo`NO-P0U=QLot|8ySUBE1=y1u!4D-dP?PX7~of^sUzM;Hrl>M6|xfHYK<rOtDn5Eul+Pc?G^9f!x}t0rkE(>Emq2Uf<VY6-eg`e&wpWx~BQ3!o|L;C|a;iO^?8q+pmsuA-w*_NqdleQLQaclj{As{(EUT%P?$IQS;}tcoIRpRYogxNm@Oj-B8SIZZImOXTN|F%Q=&9id!hhrynbrO0_`l9h-1c)_A11(6i9#D86&%!8`kNz}-8%V~(<lr{zH2Dd8xBI2n2%{*?nV#X0$+j0@ZOqu@YCo(na5J(o^AKNj3U(|eifHz3mnBPQDCJCZG37qD&|$ngM|TJa@egWgN4wBqp5sy^3XqmIE;*;AJRS)Ve2q9DVZ7Q3I`>4O~ils;whuIX3z=YA!KYQ%^<^P#*1G=_i_)i=Q6r`0i{=!pxwu?}RrgeC1@hT3<Ze9i?`0GJSHE}-qW$a;<(y#)zBY*3Hhvu){m@Ch-s`W_Jzc#*RVesEk7bj=Vi5}i<<g}BU%(*&2YtiHA%yo|D!oz>68SJ(;2=K$<Vmf|uPfjcU1FCmFZ1Js)9xzWis9#<g`xxgGUw{t7<rmLzR7jUB$0Kx!p8M2<NmQTTzNdD}ZI&UbO07bI(+&|QUfOJfCZ+s(K(ZBegkebRIAA}k@f84y}09_YI!#07$8^+>|@Co_}B}i!v3+2Ewl-n)c2f0hq00uekXOIwT-$fgG5^K17{mhtCBf>J;OMeLtDE>bE{)LjaP%Z&Iz@oLIO9~?RP|t_H=pz8ng~2`qkKP8QeC0QaZP?&*VV0xw<5Rqzc12l+YI~OC6HPS*X~D+4V5~?st^w|$YKuO+kb2P1*!bj-7#!6=n0|L&KFmj}RVNK`Yrh5uektua^xejloba;#SW9pqg4Ox5cH!&Z-l%!1)Cv7=l7pxe)nd2%=0Ms6pmHI&>s*)TZI0TrD8q7ucU{_mW*2=sHuH0yNa^f`Sucpyz51z_zhRRKJAZA`ly9n2+*o~)Kz!KmQ$z$%xR3G(_=R3yoo!P&aE-b5+kEzyq6}LdJd=Ll3H`|%a*Gz>j3PDF^5=*PkcDcTXJJHzXo#_oYZ8f1MeE|gad~uDYIU9A0=L7bm!Iw_PO91!PqI_Rtn{EJ^=T33QD0p~&xA+My1+*OdpnJ1n?!kdf^ssLpoMn?HGBh{;l%)I`a*rgh!I8&gy-l2(rH;S0tpz^t#Lp$BbPw*r<jo+S*q+o%1B#e@!8Hh(bY|MBX)V2*p!z*v3F`uc1s&xr@l^x{|q!9^??O?5kLejQ!tPwkJ7PW5g24M?|Le(XhE{JQB!8Kldyk~W0YhZlh1v*cfu*BC{NLv2`e~f!YutQ@_gq+l#JAJQJCtJz1AnY7$WvR+9+W~@eB1DxH_n2>sCJq>uRBDnkWtxw-cItpIGEhbuBzK->N9}X!F0NOlklsl+rK*?!T@Y3;Y9xU$23Tg+wlAqK<7c%;~^|QY!E7K~}|%{bfO)CfxFuL259%>((Z!_Qmi@;}lV$2G%Q4iaZi$V*%llz?>!rcqZg!+eDTzbM0@v4RZjtvs)!!;KZh6OW<y}Em(et63zUT9>P9B<-ER)8HMLv<&=emQ`xrEBs_nYYoUbj52;_|L@DA){?1`Gq#w<wpRbxV+a<(Q@lLSHx#wed{Y?Qzek<kD|8#{?LGoXrXc-Sr22jIykO5|o*s@_o$-O?(LTTYgj)wyvU5BmS+`a$Zq+jjlyP<wt+k|6tdE7tJeE9u}0oh@gqdhcT?ukE($CA!{GP;pzelGtrFPw`KmYV%!Z<zLmu@b%4wELh8JJjC^e^2Cs_{W~P%7~S;y(Oj4GeWWiKItd>F#JTE&8c;)Z0;ws&a!}S(Q(wd=jqE_FoBfJ!Vo!V35O{1$0z3e4LRPeuYoh1fnzRN(pr$&+oFa+&gpB8$5lIQbLuU2rv{W*n3)kx`i?_$Et-Mae|6yrt+G(+*wc%&7$c)K7-U~D85tGp$1%tI+<U9*6BElF#zL9)&yFa}C>HYhonDn0B~2vE;A<@J8pv&9$^3hTXOQxy4<J)4ps0xpzzVYyk(BCVU8nNc3g#Y^<{g&k5Ey@)J+{I#_cx8tU?~s+FCkb@*$VTl9~hLWN8HSX>4V$iNfY-)iZtTmE>VJ4h)I-2`c$S1-<C|$AAy-5q%h8sKv!w^E_8d*3~t)bTpZm_E!;4OWa=k|gPS@;415^U>V%&u5JfY1yZjtf_?%y^f5{yh>}e>YQ<q%sr5Np}!i7Pjw$WeDZkyuy11YI8jY6A2)x)wE9n}pdDJk`;*drh8vt&&A`(s!3MO&e7hkqPTf8Qyfw!XA|a5Icq+Ugy%q87e`3b+-QS)Ra&a0=F>;(H#yPM;On)Y7DB+KEn*bV{rfhnh6aUMG&J@sGX)<VeQlztuYrgV(6*n>`c7ECU+P08CK^d_gWc1<v)_;-Z*vP-F~US!d!8s-|U)#~sXK2vN@I#^06<3y=6{{<gDt7{^U%%x+IA^>tdP_8DX9Y2Oz#yvH-7;jkP=8>1VJnuv*KbNP$ByL>y&nHU8!&YjXpm4yYsE@UCtDGjs?@Fs~}^<oaDjXI+Gl3x}e<gic)DKQm(9x3n>8UlNo9n6t0#bvrWPDLUukCInw_N63@hNdTLLKJWEJLQ>#s;ZhE7IHHhSBSE55pnDE{mmwXg_8XKhi{+(n2cUEGVMZ5Eg{DeFzslTM=HGxpO0N8gi#t6c68mnBz25kqHbRg7|Fe!R=^NraZBhzkz9V0c2*a^imW)eye!Hs3gKGn*<^+r^|P|amcI=LL;XYbDB?pX5W_}N-f!k>v^s%z>2@nt4A&i~39+;mN}s^JE&wEY@kAvc!Mn6}Ki58dcrGy*r*Xm;Ftuw;W^W0mo@ASJABnOHVWeY}NR&l0**c~HxIwkWGW~+21epDZYJt7reaj}iDe)d_951&Ce3!!VzVO?>v}7m0E0*wO5fznuJkIU=-b@T8xU4{`pj$`fHD<}=9MriY(tR9Lu>Nzm)<Rx5Ho{$8$9BAjd!qlXIw?L)#5U820?iJeLJ=fjIm0Q_7Hq)Z2ltx7h9;uko!zv)2kEYv>OvuVU7yaOCVK1;9x_Rh4tZm7a@~upA;b@7HhVxX;auq+tLon+(ONv^oVvGRkuI5yR(%*#JfE*Bra%e&al>)z>o&W)w>nGG{U|!xUeJ}KrrA$<Q)K6vbpCpL3M?*}AYVUDt#8|4=Bttuf?g~417j$ra?A^?{Ew+Gywi`TDQHGw4)X;F4P^^!E57?lQbGz4?)yeiGxGuzF{jH`<pvoDkv5ixYQSlHr-Ro~zOuP4b_dXrHr9~rGh7{@k$ID|3=o7k&g8$^r4zvj<N-5nruIbi>+*9B;TQ9f!!`}hOgPV{DhVrFF*y`XG~eda)|boSprO~#@%54xWh+oYJ>Iaop`}*udN$z}z{Oh}0ptpC{Xp$ei59qJbv+B55yZDJZ=TWxAH(7Lx&SI+<8rEHso>0VkEa`lr&8D)Uz;cIw21rz^-Tyv0{N~(W7&Gz1;Y{nB1Lt=nDqfyx^Q`G{w6aZVSp%!ihTx0(Z)E*fcKFJW9Jr>5Pz=a6XhK3A92<`X#9(PkO{lrEsjW?3ufY~>?5PiT!hO_Ve;N!llHv*aAShm*Z>tfe480Gtgb1N55j%vzjCu46RG{?pX|S4J1MgOz5E1An+d`h!Ao};mWSodFHhsY{`vz1fZ;kWD*Znw_mC`9IggGiVpWtFq$<eA(E6bar2A6ucmMzZKW??gzP?$W00H(Upqv2!t(|7>vBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
