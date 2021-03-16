#!apps/perl/bin/perl
###########################################################################
#											       
#  - Program filename : 1_generationFV.pl	
#  - This program is mainly used to generate pairwise similarity score 
#	based on LZ complexity.
#  - The program requires New_AMPtrainingset.txt -- "Fixed Size Training Set"
#  - User need to input a argument to decide which input file they would like to select for 
#       generation pairwise feature vectors. 
#	0 - New_AMPtest.txt 		= Test sequences
#	1 - New_AMPtrain-.txt 		= General Negative Training Set
#	2 - New_AMPtrain+.txt 		= General Positive Training Set
#	example : ..\ perl 1_generationFV.pl 0
#
#	All the txt files above are containing the sequences cannot be predicted using sequence 
#	alignment method.
#	-- New_AMPtest.txt 		= Reminding Test sequences from independent test
#	-- New_AMPtrain-.txt 		= Reminding Negative Training Set from Jackknife test
#	-- New_AMPtrain+.txt 		= Reminding Positive Training Set from Jackknife test

#  - The program requires :
#     -- Fixed_AMPtrainingset.txt
#     -- New_AMPtest.txt 
#     -- New_AMPtrain-.txt or New_AMPtrain-0.7.txt
#     -- New_AMPtrain+.txt  or New_AMPtrain+0.7.txt	
#  - Output File will be named according to   FV_{inputfile}.txt
#
###########################################################################

use List::Util qw[min max];#Min and max functions are available in perl, but we need to load them first
my $start_run = time(); 

print "Generation of Feature vector is in progress.....\n";

$select =$ARGV[0] ;
@datalist =('0New_AMPtest.txt','0New_AMPtrain-.txt','0New_AMPtrain+.txt');

$datalist = @datalist[$select];
$filename="FV_$datalist";
open (FILE1, ">FV_$datalist")or die ;
open (DATA3, "<$datalist");     #read test sequences from file.
my $test = <DATA3>;
$count =0;

foreach $test(<DATA3>)      #loop for compare test set with training set.
{
	if ($test=~m/^[A-Z]/)
	{
		print "$test";
		###Assigned class label to input sequence###
		if ($datalist =~ m/\+/)
		{$sign = '+1';		
		print "fafa$ database" ;}
		elsif ($datalist =~ m/-/)
		{$sign = '-1';print "$database" ; }
		elsif ($datalist =~ m/test/)
		{$sign = '+1';print "$database" ; }

		$count=$count+1;
		@aaa=();
		$sum = 0;
		$sum1 = 0;
		###Reading training sequences from Fixed Training Set###
		open (DATA, "<Fixed_AMPtrainingset.txt");      #read positive training sequences from file.
		my $array = <DATA>;
		$test =~ s/^\s*(.*?)\s*$/$1/;                       #to eliminate any spaces in the test sequence.
		@test = split('',$test);
		###Starting to perform LZ complixity concept###
		foreach $array(<DATA>)
		{
			$array =~ s/^\s*(.*?)\s*$/$1/;    #to eliminate any spaces in the positive training sequence.	
			@array = split('',$array); #@array = Q = represent training sequence from Fixed 
			@test = split('',$test);        #@test = S =represent input sequence from inputfile
			@input1 = (@array,@test);#combine SQ 
			@input2 = (@test,@array);#combine QS
			
			##Calculation exhaustive history for Sequence Q##
			$j = 0;
			$k = 1;
			$l = 1;
			$r1= 0;
			$wq = 0;
			$b = 0;
			$lengthq =scalar(@array); # got how many element in @array = length of sequence
			for ($b =0;$b <$lengthq-1;$b++)    #loop for training
			{
				$r = 0;
				$i = 0;
				@s = @array[$i..$j];
				$seq = join("",@s);
				@q = @array[$k..$l];
				$q=join("",@q);

				if ( $seq =~ m/$q/)
				{$r=$r+1;}

				if ($b == $lengthq-2)
				{
				if ( $seq =~ m/$q/)
				{ $r1 =$r1+1;}
				}

				if($r == 0)
				{
				 $l=$l+1;
				 $j =$j+1;
				 $k =$l;
				 $wq=$wq+1;} 
				 else {$l=$l+1;$j =$j+1;}

				if($r1 > 0)
				{$wq=$wq+1;}                                                 
				}
			$answer1 =$wq+1;
			##Calculation exhaustive history for Sequence S##
			$j = 0;
			$k = 1;
			$l = 1;
			$r1= 0;
			$ws = 0;
			$b = 0;
			$lengths =scalar(@test); # got how many element in @array = length of sequence
			for ($b =0;$b <$lengths-1;$b++)  #loop for test sequence, loop to claculated the LZ complexity.
			{$r = 0;
				$i = 0;
				@s = @test[$i..$j];
				$seq = join("",@s);
				@q = @test[$k..$l];
				$q=join("",@q);

				if ( $seq =~ m/$q/)
				{$r=$r+1;}

				if ($b == $lengths-2)
				{
				if ( $seq =~ m/$q/)
				{ $r1 =$r1+1;}
				}

				if($r == 0)
				{
				 $l=$l+1;
				 $j =$j+1;
				 $k =$l;
				 $ws=$ws+1;} 
				 else {$l=$l+1;$j =$j+1;}

				if($r1 > 0)
				{$ws=$ws+1;}                                                 
				}
			$answer2 =$ws+1;    #end of loop for LZ complexity
			##Calculation exhaustive history for Sequence SQ##
			$j = 0;
			$k = 1;
			$l = 1;
			$r1= 0;
			$wqs = 0;
			$b = 0;
			$lengthqs =scalar(@input1);
			for ($b =0;$b <$lengthqs-1;$b++)
			{$r = 0;
				$i = 0;
				@s = @input1[$i..$j];
				$seq = join("",@s);
				@q = @input1[$k..$l];
				$q=join("",@q);

				if ( $seq =~ m/$q/)
				{$r=$r+1;}

				if ($b == $lengthqs-2)
				{
				if ( $seq =~ m/$q/)
				{ $r1 =$r1+1;}
				}

				if($r == 0)
				{
				 $l=$l+1;
				 $j =$j+1;
				 $k =$l;
				 $wqs=$wqs+1;} 
				 else {$l=$l+1;$j =$j+1;}

				if($r1 > 0)
				{$wqs=$wqs+1;}                                                 
				}
			$answer3 =$wqs+1;
			##Calculation exhaustive history for Sequence QS##
			$j = 0;
			$k = 1;
			$l = 1;
			$r1= 0;
			$wsq = 0;
			$b = 0;
			$lengthsq =scalar(@input2);
			for ($b =0;$b <$lengthsq-1;$b++)  #loop of q
			{$r = 0;
				$i = 0;
				@s = @input2[$i..$j];
				$seq = join("",@s);
				@q = @input2[$k..$l];
				$q=join("",@q);

				if ( $seq =~ m/$q/)
				{$r=$r+1;}

				if ($b == $lengthsq-2)
				{
				if ( $seq =~ m/$q/)
				{ $r1 =$r1+1;}
				}

				if($r == 0)
				{
				 $l=$l+1;
				 $j =$j+1;
				 $k =$l;
				 $wsq=$wsq+1;} 
				 else {$l=$l+1;$j =$j+1;}

				if($r1 > 0)
				{$wsq=$wsq+1;}                                                 
				}
			$answer4 =$wsq+1;
			## Calculating LZ similarity score between sequence S and Q## 
			$c = $answer4 - $answer2; # C(SQ)-C(S)
			$d = $answer3 - $answer1;# C(QS)-C(Q)
			$upper = max($c,$d); 
			$lower = max($answer1,$answer2);
			$dis = $upper/$lower;
			if ($test eq $array)
			{$dis=0;		}
			push @aaa,$dis;

		}
		####print pairwise simlarity score into txt file based on SVM format####
		print FILE1 "$sign	"	;

		for ($hu=0;$hu<scalar(@aaa);$hu++)
		{	
			$dis =@aaa[$hu];
			$num =$hu+1;
			print FILE1 "$num:$dis	";
		}
		print FILE1 "\n";

}
}

my $end_run = time();        # end point of calculating the elapse time.
my $run_time = $end_run - $start_run;
print "Done \n Job took $run_time seconds\n";
