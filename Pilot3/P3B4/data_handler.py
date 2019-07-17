import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from collections import Counter, OrderedDict
from sklearn.model_selection import train_test_split
#from tqdm import tqdm
import csv
import os
import shutil
import subprocess



class DataHandler():
    def __init__( self,
                  basedir = '/gpfs/alpine/proj-shared/med107/NCI_Data',
                  batch_name = 'LA_v2',
                  prep_mode = 'unify_whitespace_full',
                  data_name = 'none',
                  seed = 3545,
                  n_folds = 10,
                  data_mode = 'raw',
                  task = [ 'site', 'subsite', 'laterality', 'histology', 'behavior', 'grade' ],
                  val_perc = 0.2,
                  min_df = 5
                  ):
        self.basedir = basedir
        self.batch_name = batch_name
        self.prep_mode = prep_mode
        self.seed = seed
        self.rand_state = np.random.RandomState(seed)
        self.n_folds = n_folds
        self.data_mode = data_mode
        self.task = sorted(task)
        self.task_str = "_".join(self.task)
        self.val_perc = val_perc
        self.data_remarks = []
        self.min_df = min_df
        self.val_perc = .2
        self.seq_len = 1500
        self.reverse_tokens = True


    def get_fold( self, k, data_mode= 'cnn', task= [ 'site' ], seq_len= 1500, reverse_token= True, verbose= 1, wv_len= 300 ):
        self.seq_len = seq_len
        self.reverse_tokens = reverse_token

        datafilepath = os.path.join( self.basedir, self.prep_mode, 'Data', self.batch_name, str( self.n_folds ), str( k ) )
        picklefilename = os.path.join( datafilepath, 'cnn.pickle' )

        with open(picklefilename, 'rb') as f:
            self.truth_order = pickle.load(f)

            self.case_idx_train = pickle.load(f)
            self.case_idx_val = pickle.load(f)
            self.case_idx_test = pickle.load(f)

            self.case_label_train = pickle.load(f)
            self.case_label_val = pickle.load(f)
            self.case_label_test = pickle.load(f)

            self.train_tokens = pickle.load(f)
            self.val_tokens = pickle.load(f)
            self.test_tokens = pickle.load(f)

            self.train_y = pickle.load(f)
            self.val_y = pickle.load(f)
            self.test_y = pickle.load(f)

            self.label_encoders_array = pickle.load(f)

            self.case_xmlnames_train = pickle.load(f)
            self.case_xmlnames_val = pickle.load(f)
            self.case_xmlnames_test = pickle.load(f)

            self.labeled_vocab = pickle.load(f)

        if data_mode == 'cnn':
            self.wv_mat, self.wvToIdx = wv_initialize( self.labeled_vocab, self.rand_state, wv_len )
            self.train_y = np.array( self.train_y, dtype= 'int32' )
            self.val_y = np.array( self.val_y, dtype= 'int32' )
            self.test_y = np.array( self.test_y, dtype= 'int32' )

        task_index = []
        for t in task:
            try:
                task_index.append( self.truth_order.index( t ) )
            except:
                print('Task', t, 'not exist in the datafile.')
                print('Tasks in the datafile:', self.task)
                quit()

        task_index = np.array( task_index, dtype= 'int32' )

        self.train_y = self.train_y[ :, task_index ]
        self.val_y = self.val_y[ :, task_index ]
        self.test_y = self.test_y[ :, task_index ]

        new_label_encoder_list = []
        for i in task_index:
            new_label_encoder_list.append( self.label_encoders_array[ i ] )

        self.label_encoder_list = new_label_encoder_list

        if verbose == 1:
            print( datafilepath )
            print('==========')

            print('Number of cases:')

            print('Training set:', len(self.train_y))
            print('Validation set:', len(self.val_y))
            print('Test set:', len(self.test_y))

            print('==========')
            print('Data Stats')

            for ith_class, ith_label_encoder in enumerate( self.label_encoder_list ):
                print( "task {}: {}".format( ith_class, task[ ith_class ] ) )
                n_class = len( ith_label_encoder.classes_ )
                bins = np.zeros( ( 3, n_class ), dtype= 'int32' )

                for n in self.train_y:
                    bins[ 0, n[ ith_class ] ] = bins[ 0, n[ ith_class ] ] + 1
                for n in self.val_y:
                    bins[ 1, n[ ith_class ] ] = bins[ 1, n[ ith_class ] ] + 1
                for n in self.test_y:
                    bins[ 2, n[ ith_class ] ] = bins[ 2, n[ ith_class ] ] + 1

                print( '' )
                print( '%20s%10s%10s%10s' % ( '', 'Train', 'Val', 'Test' ) )
                for i in range( n_class ):
                    try:
                        print( '%20s%10d%10d%10d' % (
                            ith_label_encoder.inverse_transform( i ), bins[ 0, i ], bins[ 1, i ], bins[ 2, i ] ) )
                    except ValueError:
                        continue

    def make_folds( self ):
        gs3_array = []
        noYearCount = 0
        # gs3_filename = os.path.join( self.basedir, self.prep_mode, 'all_altered.csv')
        gs3_filename = os.path.join( self.basedir, 'batch2', 'gs3.csv' )
        with open( gs3_filename, 'r' ) as f:
            reader = csv.reader(f)
            first = True
            for row in reader:
                if first:
                    first = False
                    continue
                if row[3] == '' or row[3] == 'None':
                    noYearCount += 1
                    continue
                elem = []
                elem.append(row[0])
                elem.append(row[1] + ',' + row[2])
                elem.extend(row[3:])
                gs3_array.append(elem)

        print( 'Total number of reports:', len( gs3_array ) )
        print( 'Total number of reports with empty lastYearSpecCollect1:', noYearCount )
        gs3_array = np.array( gs3_array )

        case_array = gs3_array[ :, 1 ]
        year_array = gs3_array[ :, 2 ]
        year_array = [ int( float( x ) ) for x in year_array ]
        year_array = np.array( year_array )

        cv_index = np.where( ( year_array < 2016 ) & ( year_array >= 2004 ) )[ 0 ]
        final_val_index = np.where( year_array >= 2016 )[ 0 ]
        old_val_index = np.where( year_array < 2004 )[ 0 ]

        print( 'Number of reports < year 2016:', len( cv_index ) )
        print( 'Number of reports >= year 2016:', len( final_val_index ) )
        print( 'Number of reports < year 2004:', len( old_val_index ) )

        cv_cases_array = case_array[ cv_index ]
        unique_cases = np.unique( cv_cases_array )
        np.random.shuffle( unique_cases )

        study_sets = []

        N_CV = self.n_folds
        if N_CV == 0:
            train_cases, val_cases = train_test_split(unique_cases, test_size=0.2)
            train_index = case_to_index( cv_index, cv_cases_array, train_cases )
            val_index = case_to_index( cv_index, cv_cases_array, val_cases )
            test_index = np.array( final_val_index )

            elem = []
            elem.append(train_index)
            elem.append(val_index)
            elem.append(test_index)
            study_sets.append(elem)
        else:
            for fold in range(N_CV):
                tr = []
                te = []
                for k in range(len(unique_cases)):
                    if (k % N_CV) == fold:
                        te.append(unique_cases[k])
                    else:
                        tr.append(unique_cases[k])
                train_cases, val_cases = train_test_split(tr, test_size=0.2)
                train_index = case_to_index( cv_index, cv_cases_array, train_cases )
                val_index = case_to_index( cv_index, cv_cases_array, val_cases )
                test_index = case_to_index( cv_index, cv_cases_array, te )
                elem = []
                elem.append(train_index)
                elem.append(val_index)
                elem.append(test_index)
                study_sets.append(elem)
        print('Done-1')
        # with open( 'index.' + str( N_CV ) + '.pickle', 'wb') as f:
        #     pickle.dump(gs3_array, f, protocol=pickle.HIGHEST_PROTOCOL)
        #     pickle.dump(study_sets, f, protocol=pickle.HIGHEST_PROTOCOL)

        basedir = self.basedir
        batch_name = self.batch_name
        prep_mode = 'unify_whitespace_full'

        savefilename = os.path.join( self.basedir, 'batch2', self.prep_mode+'.pickle')

        with open(savefilename, 'rb') as f:
            records = pickle.load(f)

        nontext_nids = ["input_filename", "registryId", "patientIdNumber",
                        "tumorRecordNumber", "recordDocumentId"]

        labeled_records = []  # list of ([data], label)
        idxlist = []

        for record in records:
            # first, look up label
            label = 0
            # if label_file is not None:
            try:
                label = record["input_filename"]
                # print( label )
                # label = labels[ record[ "input_filename" ] ]
            except:
                continue  # unlabeled datums are ignored

            doc = sum([v for k, v in sorted(record.items(), key=lambda r: r[0])
                       if k not in nontext_nids], [])


#            CUIdir = '/mnt/nci/scratch/alawadmm/epathCUIs'
#            fname = label + '.CUI'
#            CUIfilename = os.path.join(CUIdir, fname)
#            with open(CUIfilename, 'rt') as cf:
#                body = cf.read()
#                body = body.split(' ')
#            doc = doc + body

            # labeled_records.append((doc, label))
            labeled_records.append(doc)
            idxlist.append(label)
        print('Done-2')
        labeled_records = np.array(labeled_records)
        idxlist = np.array(idxlist)


#        for row in labeled_records:
#            np.random.shuffle(row)


        seq_len = 1500
        reverse_tokens = True

        truth_order = ['site', 'subsite', 'laterality', 'histology', 'behavior', 'grade']

        datasets = []

        for f in range( len( study_sets ) ):
            # case_idx_array = []
            # case_label_array = []
            # case_tokens_array = []
            # case_y_array = []
            label_encoders_array = []
            # case_xmlnames_array = []

            for k in range(3):
                case_idx = []
                case_label = []
                case_tokens = []
                case_y = []
                label_encoders = []
                case_xmlnames = []
                #for ik in range( 100 ):
                #    i = study_sets[ f ][ k ][ ik ]
                for i in study_sets[f][k]:
                    # print( i )
                    xmlfilename = gs3_array[i, 0]
                    try:
                        c = np.where(idxlist == xmlfilename)[0][0]
                        case_idx.append(labeled_records[c])
                        # truth label
                        labels = []
                        labels.append(gs3_array[i, 3][0: 3])  # main site
                        labels.append(gs3_array[i, 3])  # subsite
                        labels.append(gs3_array[i, 4])  # laterality
                        labels.append(gs3_array[i, 7])  # histology
                        labels.append(gs3_array[i, 5])  # behavior
                        labels.append(gs3_array[i, 8])  # grade

                        case_label.append(labels)

                        # we save XML names for future reference
                        case_xmlnames.append(xmlfilename)
                    except:
                        print(xmlfilename)
                        # continue

                case_label = np.array(case_label)
                print(case_label.shape)
                print(len(case_xmlnames))
                LAcount= 0
                KYcount= 0
                for ixx in range(len(case_xmlnames)):
                    if case_xmlnames[ixx][0] == 'R':
                        LAcount += 1
                    else:
                        KYcount += 1
                print("Length of LA data xml files:", LAcount)
                print("Length of KY data xml files:", KYcount)


                if k == 0:
                    case_idx_train = np.array(case_idx)
                    case_label_train = np.array(case_label)
                    case_xmlnames_train = np.array(case_xmlnames)
                elif k == 1:
                    case_idx_val = np.array(case_idx)
                    case_label_val = np.array(case_label)
                    case_xmlnames_val = np.array(case_xmlnames)
                else:
                    case_idx_test = np.array(case_idx)
                    case_label_test = np.array(case_label)
                    case_xmlnames_test = np.array(case_xmlnames)

                # case_idx_array.append( case_idx )
                # case_label_array.append( case_label )
                # case_xmlnames_array.append( case_xmlnames )

            # case_idx_array = np.array( case_idx_array )
            # case_label_array = np.array( case_label_array )
            # case_xmlnames_array = np.array( case_xmlnames_array )

            # print( case_label_array.shape )

            case_y_train = np.array(case_label_train)
            case_y_val = np.array(case_label_val)
            case_y_test = np.array(case_label_test)

            for t in range(len(truth_order)):
                labels_train = np.unique(case_label_train[:, t].flatten())
                labels_val = np.unique(case_label_val[:, t].flatten())
                labels_test = np.unique(case_label_test[:, t].flatten())

                # labels = labels_train + labels_val + labels_test
                labels = []
                labels.extend(labels_train)
                labels.extend(labels_val)
                labels.extend(labels_test)

                label_encoder = LabelEncoder()
                label_encoder.fit(labels)

                case_y_train[:, t] = label_encoder.transform(case_y_train[:, t])
                case_y_val[:, t] = label_encoder.transform(case_y_val[:, t])
                case_y_test[:, t] = label_encoder.transform(case_y_test[:, t])

                label_encoders_array.append(label_encoder)

            # tokenize data for CNN
            min_df = 5
            labeled_vocab = vocab_handler(case_idx_train, min_df)

            savefilepath = os.path.join( self.basedir, self.prep_mode, 'Data', self.batch_name, str( self.n_folds ), str( f ) )

            print( savefilepath )

            if not os.path.exists( savefilepath ):
                os.makedirs( savefilepath )
            else:
                print( savefilepath, "remaking data_dir" )
                shutil.rmtree( savefilepath )
                os.makedirs( savefilepath )


            vocabfilename = os.path.join( savefilepath, 'vocab.pickle' )
            with open(vocabfilename, 'wb' ) as f:
                pickle.dump( labeled_vocab.vocab_idx, f, protocol= pickle.HIGHEST_PROTOCOL )

            train_tokens = [labeled_vocab.tkn_to_idx(x) for x in case_idx_train]
            val_tokens = [labeled_vocab.tkn_to_idx(x) for x in case_idx_val]
            test_tokens = [labeled_vocab.tkn_to_idx(x) for x in case_idx_test]

            train_tokens = [sequence_parcer(x, seq_len, reverse=reverse_tokens) for x in train_tokens]
            val_tokens = [sequence_parcer(x, seq_len, reverse=reverse_tokens) for x in val_tokens]
            test_tokens = [sequence_parcer(x, seq_len, reverse=reverse_tokens) for x in test_tokens]

            case_tokens_train = np.array(train_tokens, dtype='int32')
            case_tokens_val = np.array(val_tokens, dtype='int32')
            case_tokens_test = np.array(test_tokens, dtype='int32')




            picklefilename = os.path.join( savefilepath, 'cnn.pickle' )

            with open(picklefilename, 'wb') as f:
                pickle.dump(truth_order, f, protocol=pickle.HIGHEST_PROTOCOL)

                pickle.dump(case_idx_train, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(case_idx_val, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(case_idx_test, f, protocol=pickle.HIGHEST_PROTOCOL)

                pickle.dump(case_label_train, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(case_label_val, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(case_label_test, f, protocol=pickle.HIGHEST_PROTOCOL)

                pickle.dump(case_tokens_train, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(case_tokens_val, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(case_tokens_test, f, protocol=pickle.HIGHEST_PROTOCOL)

                pickle.dump(case_y_train, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(case_y_val, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(case_y_test, f, protocol=pickle.HIGHEST_PROTOCOL)

                pickle.dump(label_encoders_array, f, protocol=pickle.HIGHEST_PROTOCOL)

                pickle.dump(case_xmlnames_train, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(case_xmlnames_val, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(case_xmlnames_test, f, protocol=pickle.HIGHEST_PROTOCOL)

                pickle.dump(labeled_vocab.vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

            print( 'Change directory permission' )
            subprocess.call( [ 'chmod', '-R', '777', savefilepath ] )



def case_to_index( cv_index, case_array, cases ):
    index = []
    for i in range( len( case_array ) ):
        c = cv_index[ i ]
        if case_array[ i ] in cases:
            index.append( c )

    return index

def get_token_count(tokenlist):
    """
    function to read training data token token list
    args:
        tokenlist: list of documents as sequence of tokens
    returns:
        vocab: counter of document token occurances as ordered dict
    """
    vocab = OrderedDict()
    #set CV list indicies

    for i, tokens in enumerate(tokenlist):
        vocabCounter = dict(Counter(tokens))
        for this_token in vocabCounter.keys():
            try: vocab[this_token] += 1
            except KeyError: vocab[this_token] = 1
    return vocab

### vocab handler
class vocab_handler(object):
    """class to initialize, organize vocab
    parameters:
        - vocab: dictionary holding vocabulary/ document frequency counter
        - wv_mat: wv matrix
        - token_to_mat_idx: maps str tokens to int corresponding to wv_mat row
    """
    def __init__(self,
                 in_token_list,
                 min_df = 5):
        """ object initialation:
        - do self.vocab token count
        - initialize self.wv matrix
        - create self.token_to_mat_idx
        """
        # get vocab counter
        self.min_df = min_df
        self.max_seq_len = np.max([len(x) for x in in_token_list])
        self.token_counter = get_token_count(in_token_list)
        self.vocab = OrderedDict()
        self.vocab[""]=0
        for k,v in self.token_counter.items():
            if v>=self.min_df:
                self.vocab[k]=v
        #self.vocab = {k:v for k,v in self.token_counter.items() if v>=self.min_df}
        self.vocab.update({'<unk>':0})
        #self.vocab_idx = {k:list(self.vocab.keys()).index(k) for k,v in self.vocab.items()}
        self.vocab_idx = OrderedDict((k,list(self.vocab.keys()).index(k)) for k,v in self.vocab.items())

    def tkn_to_idx(self,in_doc):
        #converts list of tokens into a list of indices - NO PADDING
        tkn_idx_seq = []
        for word in in_doc:
            if word in self.vocab_idx:
                tkn_idx_seq.append(self.vocab_idx[word])
            else:
                tkn_idx_seq.append(self.vocab_idx['<unk>'])
        return tkn_idx_seq

def sequence_parcer(in_tkn_array,seq_len,reverse=False):
    if reverse:
        in_tkn_array = in_tkn_array[::-1]
    if len(in_tkn_array)>= seq_len:
        return in_tkn_array[:seq_len]
    else:
        return np.append(in_tkn_array,
                         np.zeros(seq_len-len(in_tkn_array),
                                  dtype = int))

def wv_initialize( train_vocab, rand_state, k= 300) :
    wordVecs = {}
    '''
    if pretrained_path != None:
        with open(pretrained_path, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                if word in vocab:
                   wordVecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.read(binary_len)
    '''
    for word in train_vocab:
        wordVecs[ word ] = rand_state.uniform( -0.25, 0.25, k )
        wordVecs[ '<unk>' ] = rand_state.uniform( -0.25, 0.25, k )

    WV_mat, wvToIdx = getidxWVs(wordVecs)

    return WV_mat,wvToIdx

def getidxWVs( loadedWV, k= 300 ):
    # Get word matrix, token to wv_mat idx mapping
    vocabSize = len( loadedWV )
    wvToIdx = dict()
    WVmatrix = np.zeros( shape= ( vocabSize + 1, k ), dtype= 'float32' )
    WVmatrix[ 0 ] = np.zeros( k, dtype= 'float32' )     #idx for padding is 0
    i = 1
    for word in loadedWV:
        WVmatrix[ i ] = loadedWV[ word ]
        wvToIdx[ word ] = i
        i += 1
    return WVmatrix, wvToIdx



def main():
    makeNewData = True
    if makeNewData:
        data_handler = DataHandler( n_folds= 2, prep_mode= 'basic' )
        data_handler.make_folds()
    else:
        data_handler = DataHandler(n_folds=2, data_name='package2',
                                   task=[ 'site', 'subsite', 'laterality', 'behavior', 'histology', 'grade' ] )
        data_handler.get_fold( 0, data_mode='cnn', task=['site', 'subsite', 'laterality', 'behavior', 'histology', 'grade'])
        # print( ret )
    # data_handler.task = [ 'type', 'laterality', 'histology', 'behavior', 'grade' ]
    # data_handler.make_fold( 0, csv_dir_path = '/lustre/atlas/proj-shared/csc264/NCI_Data/filename_to_csv/GL2' )


if __name__ == "__main__":
    main()

