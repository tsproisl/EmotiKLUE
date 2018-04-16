# Some baselines for IEST #

## Majority baseline ##

Labels: fear;surprise;disgust;joy;anger;sad
Label	TP	FP	FN	P	R	F
fear	0	0	1600	1.0	0.0	0.0
surprise	0	0	1600	1.0	0.0	0.0
disgust	0	0	1600	1.0	0.0	0.0
joy	0	0	1600	1.0	0.0	0.0
anger	1600	8000	0	0.167	1.0	0.286
sad	0	0	1600	1.0	0.0	0.0
MicAvg	1600	8000	8000	0.167	0.167	0.167
MacAvg				0.861	0.167	0.048
Official result: 0.047619047619047616

## Bag of words (Naive Bayes) ##

Labels: fear;surprise;disgust;joy;anger;sad
Label	TP	FP	FN	P	R	F
fear	820	826	780	0.498	0.512	0.505
surprise	741	889	859	0.455	0.463	0.459
disgust	895	771	705	0.537	0.559	0.548
joy	810	777	790	0.51	0.506	0.508
anger	632	749	968	0.458	0.395	0.424
sad	694	996	906	0.411	0.434	0.422
MicAvg	4592	5008	5008	0.478	0.478	0.478
MacAvg				0.478	0.478	0.478
Official result: 0.47772501554699626

## Bag of words tf-idf (Naive Bayes) ##

Labels: fear;surprise;disgust;joy;anger;sad
Label	TP	FP	FN	P	R	F
fear	817	841	783	0.493	0.511	0.502
surprise	645	787	955	0.45	0.403	0.425
disgust	898	873	702	0.507	0.561	0.533
joy	776	786	824	0.497	0.485	0.491
anger	603	741	997	0.449	0.377	0.41
sad	698	1135	902	0.381	0.436	0.407
MicAvg	4437	5163	5163	0.462	0.462	0.462
MacAvg				0.463	0.462	0.461
Official result: 0.4611487943389731

## Bag of words (Linear SVC) ##

Labels: fear;surprise;disgust;joy;anger;sad
Label	TP	FP	FN	P	R	F
fear	816	793	784	0.507	0.51	0.509
surprise	726	893	874	0.448	0.454	0.451
disgust	887	869	713	0.505	0.554	0.529
joy	805	820	795	0.495	0.503	0.499
anger	653	827	947	0.441	0.408	0.424
sad	622	889	978	0.412	0.389	0.4
MicAvg	4509	5091	5091	0.47	0.47	0.47
MacAvg				0.468	0.47	0.469
Official result: 0.46856151603440677

## Bag of words tf-idf (Linear SVC) ##

Labels: fear;surprise;disgust;joy;anger;sad
Label	TP	FP	FN	P	R	F
fear	855	787	745	0.521	0.534	0.527
surprise	764	906	836	0.457	0.477	0.467
disgust	919	817	681	0.529	0.574	0.551
joy	833	787	767	0.514	0.521	0.517
anger	678	827	922	0.45	0.424	0.437
sad	619	808	981	0.434	0.387	0.409
MicAvg	4668	4932	4932	0.486	0.486	0.486
MacAvg				0.484	0.486	0.485
Official result: 0.48479696413185175

## Bag of uni- and bigrams (Linear SVC) ##

Labels: fear;surprise;disgust;joy;anger;sad
Label	TP	FP	FN	P	R	F
fear	919	650	681	0.586	0.574	0.58
surprise	872	795	728	0.523	0.545	0.534
disgust	914	687	686	0.571	0.571	0.571
joy	917	709	683	0.564	0.573	0.569
anger	757	768	843	0.496	0.473	0.484
sad	773	839	827	0.48	0.483	0.481
MicAvg	5152	4448	4448	0.537	0.537	0.537
MacAvg				0.537	0.537	0.537
Official result: 0.5365323746236651

## Bag of uni- and bigrams tf-idf (Linear SVC) ##

Labels: fear;surprise;disgust;joy;anger;sad
Label	TP	FP	FN	P	R	F
fear	1000	630	600	0.613	0.625	0.619
surprise	933	697	667	0.572	0.583	0.578
disgust	965	601	635	0.616	0.603	0.61
joy	952	691	648	0.579	0.595	0.587
anger	825	740	775	0.527	0.516	0.521
sad	812	754	788	0.519	0.507	0.513
MicAvg	5487	4113	4113	0.572	0.572	0.572
MacAvg				0.571	0.572	0.571
Official result: 0.571315642531662
