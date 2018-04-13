# Some baselines for IEST #

## Majority baseline ##

Labels: surprise;sad;anger;joy;disgust;fear
Label	TP	FP	FN	P	R	F
surprise	1600	8000	0	0.167	1.0	0.286
sad	0	0	1600	1.0	0.0	0.0
anger	0	0	1600	1.0	0.0	0.0
joy	0	0	1600	1.0	0.0	0.0
disgust	0	0	1600	1.0	0.0	0.0
fear	0	0	1600	1.0	0.0	0.0
MicAvg	1600	8000	8000	0.167	0.167	0.167
MacAvg				0.861	0.167	0.048
Official result: 0.047619047619047616

## Bag of words (Naive Bayes) ##

Labels: surprise;sad;anger;joy;disgust;fear
Label	TP	FP	FN	P	R	F
surprise	741	889	859	0.455	0.463	0.459
sad	694	996	906	0.411	0.434	0.422
anger	632	749	968	0.458	0.395	0.424
joy	810	777	790	0.51	0.506	0.508
disgust	895	771	705	0.537	0.559	0.548
fear	820	826	780	0.498	0.512	0.505
MicAvg	4592	5008	5008	0.478	0.478	0.478
MacAvg				0.478	0.478	0.478
Official result: 0.47772501554699626

## Bag of words tf-idf (Naive Bayes) ##

Labels: surprise;sad;anger;joy;disgust;fear
Label	TP	FP	FN	P	R	F
surprise	645	787	955	0.45	0.403	0.425
sad	698	1135	902	0.381	0.436	0.407
anger	603	741	997	0.449	0.377	0.41
joy	776	786	824	0.497	0.485	0.491
disgust	898	873	702	0.507	0.561	0.533
fear	817	841	783	0.493	0.511	0.502
MicAvg	4437	5163	5163	0.462	0.462	0.462
MacAvg				0.463	0.462	0.461
Official result: 0.46114879433897316

## Bag of words (Linear SVC) ##

Labels: surprise;sad;anger;joy;disgust;fear
Label	TP	FP	FN	P	R	F
surprise	725	888	875	0.449	0.453	0.451
sad	622	889	978	0.412	0.389	0.4
anger	654	827	946	0.442	0.409	0.425
joy	806	825	794	0.494	0.504	0.499
disgust	886	869	714	0.505	0.554	0.528
fear	816	793	784	0.507	0.51	0.509
MicAvg	4509	5091	5091	0.47	0.47	0.47
MacAvg				0.468	0.47	0.469
Official result: 0.46855897442165045

## Bag of words tf-idf (Linear SVC) ##

Labels: surprise;sad;anger;joy;disgust;fear
Label	TP	FP	FN	P	R	F
surprise	764	906	836	0.457	0.477	0.467
sad	619	808	981	0.434	0.387	0.409
anger	678	827	922	0.45	0.424	0.437
joy	833	787	767	0.514	0.521	0.517
disgust	919	817	681	0.529	0.574	0.551
fear	855	787	745	0.521	0.534	0.527
MicAvg	4668	4932	4932	0.486	0.486	0.486
MacAvg				0.484	0.486	0.485
Official result: 0.48479696413185186
