#include "SLUMTsolver.h"

EigenVec SLUMTSolver::solve_refact(const EigenMat & eigenA,
                                   const EigenVec & eigen_rhs )
{
    using namespace SuperLU;

    auto K_copy = eigenA;
    auto rhs_copy = eigen_rhs;

    SuperMatrix B, X;

    int_t         nprocs;
    fact_t      fact;
    trans_t     trans;
    yes_no_t    refact, usepr;
    equed_t     equed;

    int_t         info, lwork, nrhs, ldx, panel_size, relax;

    int_t         m, n, nnz, permc_spec;
    int_t         i, firstfact;

    double      drop_tol, rpg, rcond;
    superlu_memusage_t superlu_memusage;

    char *numthreads_env = getenv("GLFF_NUMTHREADS");
    if(numthreads_env==NULL)
    {
        nprocs = 2;
    }
    else
    {
        nprocs = atoi(numthreads_env);
    }
    printf("Using %d threads\n",nprocs);

    fact  = EQUILIBRATE;
    trans = NOTRANS;
    equed = NOEQUIL;
    refact= NO;
    panel_size = sp_ienv(1);
    relax = sp_ienv(2);

    usepr = NO;
    drop_tol = 0.0;
    lwork = 0;

    nrhs  = 1;

    n = m = eigenA.rows();

    EigenVec result(n);

    dCreate_CompCol_Matrix(&A, m, n, eigenA.nonZeros(), (double*)eigenA.valuePtr(),
                           (int*)eigenA.innerIndexPtr(),
                           (int*)eigenA.outerIndexPtr(), SLU_NC, SLU_D, SLU_GE);

    NCformat *Astore = (NCformat*)A.Store;
    printf("Dimension " IFMT "x" IFMT "; # nonzeros " IFMT "\n", A.nrow, A.ncol, Astore->nnz);

    if(this->factored == false)
    {
        if (!(this->perm_r = intMalloc(m))) SUPERLU_ABORT("Malloc fails for perm_r[].");
        if (!(this->perm_c = intMalloc(n))) SUPERLU_ABORT("Malloc fails for perm_c[].");
        if ( !(this->R = (double *) SUPERLU_MALLOC(A.nrow * sizeof(double))) )
            SUPERLU_ABORT("SUPERLU_MALLOC fails for R[].");
        if ( !(this->C = (double *) SUPERLU_MALLOC(A.ncol * sizeof(double))) )
            SUPERLU_ABORT("SUPERLU_MALLOC fails for C[].");

        /*
        * Get column permutation vector perm_c[], according to permc_spec:
        *   permc_spec = 0: natural ordering
        *   permc_spec = 1: minimum degree ordering on structure of A'*A
        *   permc_spec = 2: minimum degree ordering on structure of A'+A
        *   permc_spec = 3: approximate minimum degree for unsymmetric matrices
        */

        permc_spec = 2;
        get_perm_c(permc_spec, &A, perm_c);

        if ( !(superlumt_options.etree = intMalloc(n)) )
            SUPERLU_ABORT("Malloc fails for etree[].");
        if ( !(superlumt_options.colcnt_h = intMalloc(n)) )
            SUPERLU_ABORT("Malloc fails for colcnt_h[].");
        if ( !(superlumt_options.part_super_h = intMalloc(n)) )
            SUPERLU_ABORT("Malloc fails for colcnt_h[].");

        if ( !(ferr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) )
            SUPERLU_ABORT("SUPERLU_MALLOC fails for ferr[].");
        if ( !(berr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) )
            SUPERLU_ABORT("SUPERLU_MALLOC fails for berr[].");

        this->superlumt_options.SymmetricMode = YES;
        this->superlumt_options.diag_pivot_thresh = 0.0;

        this->superlumt_options.nprocs = nprocs;
        this->superlumt_options.fact = fact;
        this->superlumt_options.trans = trans;
        this->superlumt_options.refact = refact;
        this->superlumt_options.panel_size = panel_size;
        this->superlumt_options.relax = relax;
        this->superlumt_options.usepr = usepr;
        this->superlumt_options.drop_tol = drop_tol;
        this->superlumt_options.PrintStat = NO;
        this->superlumt_options.perm_c = this->perm_c;
        this->superlumt_options.perm_r = this->perm_r;
        this->superlumt_options.lwork = 0;
        this->superlumt_options.work = NULL;

    }
    else
    {
        this->superlumt_options.refact = YES;
    }

    dCreate_Dense_Matrix(&B, m, 1, (double*) eigen_rhs.data(), m, SLU_DN, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&X, m, 1, (double*) result.data(), m, SLU_DN, SLU_D, SLU_GE);

    printf("sym_mode %d\tdiag_pivot_thresh %.4e\n",
            this->superlumt_options.SymmetricMode,
            this->superlumt_options.diag_pivot_thresh);

    puts("LS config");

    pdgssvx(nprocs, &this->superlumt_options, &A, perm_c, perm_r,
            &equed, this->R, this->C, &this->L, &this->U, &B, &X, &rpg, &rcond,
            ferr, berr, &superlu_memusage, &info);

    printf("psgssvx(): info " IFMT "\n", info);

    if(!this->factored)
        this->factored = true;



    if ( info == 0 || info == n+1 ) {

	printf("Recip. pivot growth = %e\n", rpg);
	printf("Recip. condition number = %e\n", rcond);
	printf("%8s%16s%16s\n", "rhs", "FERR", "BERR");
	for (i = 0; i < nrhs; ++i) {
	    printf(IFMT "%16e%16e\n", i+1, ferr[i], berr[i]);
	}

    auto Lstore = (SCPformat *) this->L.Store;
    auto Ustore = (NCPformat *) this->U.Store;
	printf("No of nonzeros in factor L = " IFMT "\n", Lstore->nnz);
    	printf("No of nonzeros in factor U = " IFMT "\n", Ustore->nnz);
    	printf("No of nonzeros in L+U = " IFMT "\n", Lstore->nnz + Ustore->nnz - n);
	printf("L\\U MB %.3f\ttotal MB needed %.3f\texpansions " IFMT "\n",
	       superlu_memusage.for_lu/1e6, superlu_memusage.total_needed/1e6,
	       superlu_memusage.expansions);

	fflush(stdout);

    } else if ( info > 0 && lwork == -1 ) {
        printf("** Estimated memory: " IFMT " bytes\n", info - n);
    }

    // Only need to free structure part, instead of actual data
    // (which have been stored in Eigen objects)

    SUPERLU_FREE(A.Store);
    SUPERLU_FREE(B.Store);
    SUPERLU_FREE(X.Store);

    return result;
}

/*
Eigen::Matrix<double,1,-1,Eigen::RowMajor>
    SPLUMT_solve(const Eigen::SparseMatrix<double> &eigen_K,
                 const Eigen::Matrix<double,1,Eigen::Dynamic,Eigen::RowMajor> & eigen_rhs)
{
    using namespace SuperLU;
    auto K_copy = eigen_K;
    auto rhs_copy = eigen_rhs;

    SuperMatrix A, L, U;
    SuperMatrix B, X;
    NCformat    *Astore;
    SCPformat   *Lstore;
    NCPformat   *Ustore;
    int_t         nprocs;
    fact_t      fact;
    trans_t     trans;
    yes_no_t    refact, usepr;
    equed_t     equed;
    double      *a;
    int_t         *asub, *xa;
    int_t         *perm_c; // column permutation vector
    int_t         *perm_r; // row permutations from partial pivoting
    void        *work;
    superlumt_options_t superlumt_options;
    int_t         info, lwork, nrhs, ldx, panel_size, relax;
    int_t         m, n, nnz, permc_spec;
    int_t         i, firstfact;
    double      *rhsb, *rhsx, *xact;
    double      *R, *C;
    double      *ferr, *berr;
    double      u, drop_tol, rpg, rcond;
    superlu_memusage_t superlu_memusage;

    nprocs = 4;

    fact  = EQUILIBRATE;
    trans = NOTRANS;
    equed = NOEQUIL;
    refact= NO;
    panel_size = sp_ienv(1);
    relax = sp_ienv(2);
    u     = 1.0;
    usepr = NO;
    drop_tol = 0.0;
    lwork = 0;
    nrhs  = 1;

    n = m = eigen_K.rows();

    Eigen::Matrix<double,1,-1,Eigen::RowMajor> result(n);

    puts("LSP1");

    dCreate_CompCol_Matrix(&A, m, n, eigen_K.nonZeros(), (double*)eigen_K.valuePtr(),
                           (int*)eigen_K.innerIndexPtr(),
                           (int*)eigen_K.outerIndexPtr(), SLU_NC, SLU_D, SLU_GE);

    Astore = (NCformat*)A.Store;
    printf("Dimension " IFMT "x" IFMT "; # nonzeros " IFMT "\n", A.nrow, A.ncol, Astore->nnz);


    firstfact = (fact == FACTORED || refact == YES);
    puts("LSP2");

    dCreate_Dense_Matrix(&B, m, 1, (double*) eigen_rhs.data(), m, SLU_DN, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&X, m, 1, (double*) result.data(), m, SLU_DN, SLU_D, SLU_GE);
    puts("LSP3");

    if (!(perm_r = intMalloc(m))) SUPERLU_ABORT("Malloc fails for perm_r[].");
    if (!(perm_c = intMalloc(n))) SUPERLU_ABORT("Malloc fails for perm_c[].");
    if ( !(R = (double *) SUPERLU_MALLOC(A.nrow * sizeof(double))) )
        SUPERLU_ABORT("SUPERLU_MALLOC fails for R[].");
    if ( !(C = (double *) SUPERLU_MALLOC(A.ncol * sizeof(double))) )
        SUPERLU_ABORT("SUPERLU_MALLOC fails for C[].");
    if ( !(ferr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) )
        SUPERLU_ABORT("SUPERLU_MALLOC fails for ferr[].");
    if ( !(berr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) )
        SUPERLU_ABORT("SUPERLU_MALLOC fails for berr[].");

     * Get column permutation vector perm_c[], according to permc_spec:
     *   permc_spec = 0: natural ordering
     *   permc_spec = 1: minimum degree ordering on structure of A'*A
     *   permc_spec = 2: minimum degree ordering on structure of A'+A
     *   permc_spec = 3: approximate minimum degree for unsymmetric matrices

    puts("LSP4");

    permc_spec = 2;
    get_perm_c(permc_spec, &A, perm_c);

    puts("LSP5");

    superlumt_options.SymmetricMode = YES;
    superlumt_options.diag_pivot_thresh = 0.0;

    superlumt_options.nprocs = nprocs;
    superlumt_options.fact = fact;
    superlumt_options.trans = trans;
    superlumt_options.refact = refact;
    superlumt_options.panel_size = panel_size;
    superlumt_options.relax = relax;
    superlumt_options.usepr = usepr;
    superlumt_options.drop_tol = drop_tol;
    superlumt_options.PrintStat = NO;
    superlumt_options.perm_c = perm_c;
    superlumt_options.perm_r = perm_r;
    superlumt_options.work = work;
    superlumt_options.lwork = lwork;
    if ( !(superlumt_options.etree = intMalloc(n)) )
	SUPERLU_ABORT("Malloc fails for etree[].");
    if ( !(superlumt_options.colcnt_h = intMalloc(n)) )
	SUPERLU_ABORT("Malloc fails for colcnt_h[].");
    if ( !(superlumt_options.part_super_h = intMalloc(n)) )
	SUPERLU_ABORT("Malloc fails for colcnt_h[].");

    printf("sym_mode %d\tdiag_pivot_thresh %.4e\n",
	   superlumt_options.SymmetricMode,
	   superlumt_options.diag_pivot_thresh);

    puts("LS config");

    pdgssvx(nprocs, &superlumt_options, &A, perm_c, perm_r,
	    &equed, R, C, &L, &U, &B, &X, &rpg, &rcond,
	    ferr, berr, &superlu_memusage, &info);

    printf("psgssvx(): info " IFMT "\n", info);

    puts("LSP5");



    if ( info == 0 || info == n+1 ) {

	printf("Recip. pivot growth = %e\n", rpg);
	printf("Recip. condition number = %e\n", rcond);
	printf("%8s%16s%16s\n", "rhs", "FERR", "BERR");
	for (i = 0; i < nrhs; ++i) {
	    printf(IFMT "%16e%16e\n", i+1, ferr[i], berr[i]);
	}

        Lstore = (SCPformat *) L.Store;
        Ustore = (NCPformat *) U.Store;
	printf("No of nonzeros in factor L = " IFMT "\n", Lstore->nnz);
    	printf("No of nonzeros in factor U = " IFMT "\n", Ustore->nnz);
    	printf("No of nonzeros in L+U = " IFMT "\n", Lstore->nnz + Ustore->nnz - n);
	printf("L\\U MB %.3f\ttotal MB needed %.3f\texpansions " IFMT "\n",
	       superlu_memusage.for_lu/1e6, superlu_memusage.total_needed/1e6,
	       superlu_memusage.expansions);

	fflush(stdout);

    } else if ( info > 0 && lwork == -1 ) {
        printf("** Estimated memory: " IFMT " bytes\n", info - n);
    }


    SUPERLU_FREE (rhsb);
    SUPERLU_FREE (rhsx);
    SUPERLU_FREE (xact);
    SUPERLU_FREE (perm_r);
    SUPERLU_FREE (perm_c);
    SUPERLU_FREE (R);
    SUPERLU_FREE (C);
    SUPERLU_FREE (ferr);
    SUPERLU_FREE (berr);

    //Destroy_CompCol_Matrix(&A);
    //Destroy_SuperMatrix_Store(&B);
    //Destroy_SuperMatrix_Store(&X);

    SUPERLU_FREE (superlumt_options.etree);
    SUPERLU_FREE (superlumt_options.colcnt_h);
    SUPERLU_FREE (superlumt_options.part_super_h);
    if ( lwork == 0 ) {
        Destroy_SuperNode_SCP(&L);
        Destroy_CompCol_NCP(&U);
    } else if ( lwork > 0 ) {
        SUPERLU_FREE(work);
    }

    puts("EEEE");

    return result;
}
*/


