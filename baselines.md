# Some baselines for IEST #

## Majority baseline ##

Labels: sad;disgust;surprise;joy;fear;anger
Label	TP	FP	FN	P	R	F
sad	0	0	1460	1.0	0.0	0.0
disgust	0	0	1597	1.0	0.0	0.0
surprise	0	0	1600	1.0	0.0	0.0
joy	1736	7855	0	0.181	1.0	0.307
fear	0	0	1598	1.0	0.0	0.0
anger	0	0	1600	1.0	0.0	0.0
MicAvg	1736	7855	7855	0.181	0.181	0.181
MacAvg				0.864	0.167	0.051
Official result: 0.05108737235513964

## Bag of words (Naive Bayes) ##

Labels: sad;disgust;surprise;joy;fear;anger
Label	TP	FP	FN	P	R	F
sad	550	707	910	0.438	0.377	0.405
disgust	876	781	721	0.529	0.549	0.538
surprise	738	911	862	0.448	0.461	0.454
joy	1104	872	632	0.559	0.636	0.595
fear	819	830	779	0.497	0.513	0.504
anger	640	763	960	0.456	0.4	0.426
MicAvg	4727	4864	4864	0.493	0.493	0.493
MacAvg				0.488	0.489	0.487
Official result: 0.48718331013903365

## Bag of words tf-idf (Naive Bayes) ##

Labels: sad;disgust;surprise;joy;fear;anger
Label	TP	FP	FN	P	R	F
sad	371	388	1089	0.489	0.254	0.334
disgust	905	899	692	0.502	0.567	0.532
surprise	625	804	975	0.437	0.391	0.413
joy	1202	1395	534	0.463	0.692	0.555
fear	807	846	791	0.488	0.505	0.496
anger	604	745	996	0.448	0.378	0.41
MicAvg	4514	5077	5077	0.471	0.471	0.471
MacAvg				0.471	0.464	0.457
Official result: 0.4566939401401855

## Bag of words (Linear SVC) ##

Labels: sad;disgust;surprise;joy;fear;anger
Label	TP	FP	FN	P	R	F
sad	606	816	854	0.426	0.415	0.421
disgust	881	856	716	0.507	0.552	0.528
surprise	721	881	879	0.45	0.451	0.45
joy	1021	714	715	0.588	0.588	0.588
fear	817	798	781	0.506	0.511	0.509
anger	646	834	954	0.436	0.404	0.419
MicAvg	4692	4899	4899	0.489	0.489	0.489
MacAvg				0.486	0.487	0.486
Official result: 0.4859536180855588

## Bag of words tf-idf (Linear SVC) ##

Labels: sad;disgust;surprise;joy;fear;anger
Label	TP	FP	FN	P	R	F
sad	586	737	874	0.443	0.401	0.421
disgust	910	812	687	0.528	0.57	0.548
surprise	760	906	840	0.456	0.475	0.465
joy	1044	673	692	0.608	0.601	0.605
fear	866	778	732	0.527	0.542	0.534
anger	684	835	916	0.45	0.427	0.439
MicAvg	4850	4741	4741	0.506	0.506	0.506
MacAvg				0.502	0.503	0.502
Official result: 0.5020698555771235

## Bag of uni- and bigrams (Linear SVC) ##

Labels: sad;disgust;surprise;joy;fear;anger
Label	TP	FP	FN	P	R	F
sad	732	698	728	0.512	0.501	0.507
disgust	915	708	682	0.564	0.573	0.568
surprise	871	779	729	0.528	0.544	0.536
joy	1147	608	589	0.654	0.661	0.657
fear	937	673	661	0.582	0.586	0.584
anger	758	765	842	0.498	0.474	0.485
MicAvg	5360	4231	4231	0.559	0.559	0.559
MacAvg				0.556	0.557	0.556
Official result: 0.5562684907003398

## Bag of uni- and bigrams tf-idf (Linear SVC) ##

Labels: sad;disgust;surprise;joy;fear;anger
Label	TP	FP	FN	P	R	F
sad	796	641	664	0.554	0.545	0.55
disgust	957	625	640	0.605	0.599	0.602
surprise	942	702	658	0.573	0.589	0.581
joy	1175	546	561	0.683	0.677	0.68
fear	1001	637	597	0.611	0.626	0.619
anger	822	747	778	0.524	0.514	0.519
MicAvg	5693	3898	3898	0.594	0.594	0.594
MacAvg				0.592	0.592	0.592
Official result: 0.5915992378821602
