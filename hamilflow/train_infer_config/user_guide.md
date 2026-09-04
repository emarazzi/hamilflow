DeepH-pack User Guide

                                                   Welcome to the new version of DeepH!

                                                              Prepared by: Yang Li, Yanzhen Wang and Boheng Zhao
                                                                                                        Date: Jan 05, 2026(v1.0.6)
                                                                                                                       Visit: CMT.group

I GETTING STARTED 3 1 2 INTRODUCTION & SOFTWARE HISTORY 4 3 3.1 OBTAIN
THE PACKAGE 5 3.2 INSTALLATION GUIDE 6 II DATA PREPARATION 4 Environment
setup 6 4.1 4.2 Install DeepH-pack 6 4.3 5 7 5.1 5.2 DFT DATA GENERATION
8 6 6.1 New version of DeepH-pack DFT Data 8

III TRAINING MODELS Using DeepH-dock to Prepare DFT Data 9 7 7.1 Notes
on DFT calculations 10 7.2 8 BUILDING THE DATA GRAPH 11 8.1 9 What is
the Graph File 11 9.1 9.2 Building the Graph 12

                                    DATA ANALYSIS                         13

                                    Analysis of Dataset Features          13

                                                                          15

                                    USER CONFIGURATION                    16

                                    Training                              16

                                    Inference                             30

                                    TRAINING AND INFERENCE                34

                                    Run DeepH                             34

                                    AFTER TRAINING AND INFERENCE          35

                                    After Training                        35

                                    Data Post-Processing                  35

GETTING STARTED

                                                                                                                                           3

GETTING STARTED 1. INTRODUCTION & SOFTWARE HISTORY

                       DeepH-pack embodies the cumulative efforts of successive researchers from AI Physics
                       Research Center ,Department of Physics, Tsinghua University. Having undergone rigor-
                       ous long-term testing across all neural modules, DeepH-pack now achieves comprehensive
                       maturity. The current version features a JAX-based reconstruction with static computa-
                       tional graphs and algorithmic advancements. Looking forward, DeepH-pack’s development
                       roadmap envisions seamless integration of multi-framework backends, evolving into an ex-
                       tensible computational ecosystem for quantum materials modeling. This strategic upgrade
                       will establish cross-platform compatibility while preserving the toolkit’s signature accuracy
                       in Hamiltonian construction.
                       Benefiting from the significant computational optimizations enabled by JAX static graphs
                       and latest innovative neural network algorithms, the current DeepH-pack demonstrates ex-
                       ceptional performance in runtime, precision, and memory eﬀiciency. We warmly welcome
                       any feedback while dedicating ourselves to constructing an expanded and more compre-
                       hensive DeepH-pack platform for materials computation and predictive modeling.
                       In addition to this tutorial manual, we successfully hosted the inaugural DeepH-pack
                       Global User Conference in December 2025, bringing together a wide range of users from
                       both academia and industry. The tutorial materials and practical cases have been sys-
                       tematically organized and are now available as permanent online resources at https:
                      //deeph-workshop-tutorial.readthedocs.io/en/latest/ for on-demand access and
                       learning. This repository will be continuously updated, aiming to provide users with an
                       open learning platform that covers everything from basics to advanced applications. We
                       cordially invite all users to make full use of these resources and join the community in
                       jointly advancing the development of quantum materials computation and artificial intel-
                       ligence methods.

                                                                                                                                           4

GETTING STARTED 2. OBTAIN THE PACKAGE

                        The DeepH software package is currently distributed as an encrypted archive. To obtain
                        it, interested researchers are kindly invited to submit a request through DeepH Team’s
                        dedicated collaboration portal at https://ticket.deeph-pack.com. This process helps
                        facilitate academic collaboration while adhering to institutional intellectual property guide-
                        lines and open science principles.
                        Please note that the core DeepH training code is still under active development and is
                        not yet open source. However, to foster community engagement and application devel-
                        opment, all external interfaces for performing computations with trained DeepH mod-
                        els are fully open source. We have established the DeepH-dock project (available at
                        https://github.com/kYangLi/DeepH-dock), which provides these open-source inference
                        interfaces and tools. We warmly welcome the community to explore the repository, use
                        the interfaces, and contribute to its development.

                                                                                                                                            5

3. INSTALLATION GUIDE

3.1. ENVIRONMENT SETUP

        The DeepH-pack installation follows a streamlined procedure with robust dependency man-
         agement. The package is built via uv, a fast and versatile python package manager. For
         optimal performance, we oﬀicially recommend the following steps:
        • Install uv:

            https://docs.astral.sh/uv/#installation
        • Create python 3.13 environment via uv:

              $ uv venv deeph --python=3.13
              $ source deeph/bin/activate

        • We recommend configuring high-performance mirrors based on your IP location. For
            example, for users in China, we recommend the following mirror:
            https://pypi.tuna.tsinghua.edu.cn/simple
            You can add the mirror by:

              $ echo 'export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"'
                     >> ~/.bashrc

        The CUDA 12.9/13.1 is also needed if you have NVIDIA GPU in your machine.

3.2. INSTALL DEEPH-PACK

         Once the package access petition is granted, you will receive a wheel named
         deepx-1.0.6+light-py3-none-any.whl, which constitutes an encrypted and compressed
         archive. You can install the DeepH-pack and all the required dependencies into the uv
         environment by the following command (if you are working on a GPU platform):

           $ uv pip install ./deepx -1.0.6+light-py3-none-any.whl[gpu] --extra-index-url
                 https://download.pytorch.org/whl/cpu

        The tag ”gpu” corresponds to gpu version of the site packages such as jax, flax, and
         optax. If you are working on a CPU-only platform, use tag ”cpu” instead.
        You can check whether the DeepH-pack is successfully installed:

           $ python
           $ import deepx
           $ deepx.__version__

        The installation is successful if you see the output '1.0.6'.

                                                                                                                             6

DATA PREPARATION

                                                                                                                                           7

4. DFT DATA GENERATION

                  4.1. NEW VERSION OF DEEPH-PACK DFT DATA

                  In the latest version of DeepH-pack, we have adopted a new folder layout that is lighter,
                  more user-friendly, and with higher IO throughput. The folder tree structure looks like
                  this:

                  dft
                     |- 0
                           |- POSCAR
                           |- info.json
                           |- overlap.h5
                           |- hamiltonian.h5 (optional)
                           |- density_matrix.h5 (optional)
                     |- 1
                     |- ...

                  where overlap.h5, hamiltonian.h5 and density_matrix.h5 are the corresponding ma-
                  trices under localized atomic orbital (AO) basis. The root directory for DFT raw data
                  must be strictly named dft/, while the subfolders inside can be named more freely with
                  some naming conventions (e.g., free-form labels instead of numerical indices like 0, 1,
                  2...).The info.json contains the overall information of the system, regardless of the spe-
                  cific geometric structure. A tyipical info.json file looks like this:

                  {
                     "atoms_quantity": 75, # Number of atoms in the unit cell
                     "orbits_quantity": 1125, # Total number of orbitals in the unit cell
                     "orthogonal_basis": false, # Whether the orbitals are orthogonal
                     "spinful": false, # Whether the system is spinful , true for SOC systems
                     "fermi_energy_eV": -4.966579608421203, # Fermi energy in eV
                     "elements_orbital_map": {
                         "Mo": [0, 0, 0, 1, 1, 2, 2], "S": [0, 0, 1, 1, 2]
                     } # Angular momentum of orbitals for each element

                  }

DATA PREPARATION The \*.h5 file is a sparse storage of the physical
properties in real space and contains four parts.

                  1 $ python

                  2 >>> import h5py

                  3 >>> with h5py.File('overlap.h5', 'r') as f:

                  4 ...       print(list(f.keys()))

                  5 ...

                  6 ['atom_pairs', 'chunk_boundaries', 'chunk_shapes', 'entries']

                  The atom_pairs is a N_edge × 5 array storing the edges (or “hoppings” in physical
                  perspective), with each line takes the form [R1, R2, R3, i, j]. The R1,2,3 denote the
                  relative lattice shift along three lattice vectors from the start atom to the end atom. The i
                  and j denote the index of the start atom and the end atom, with the indexes starting from
                  0 and corresponding to the order of atoms specified in POSCAR. The entries is a 1-D array
                  containing all the matrix elements of the edges recorded in the atom_pairs. The blocks
                  Ai,j,R are flattened and concatenated to form the entries object. The chunk_boundaries
                  is a N_edge+1 array recording the split indexes of the blocks in the entries object. The
                  chunk_shapes is a N_edge × 2 array recording the shapes of the blocks. The matrix

                                                                                                                                      8

form of the physical property can be recovered by splitting and
reshaping the entries object according to the chunk_boundaries and
chunk_shapes. The order of the orbitals follows the elements_orbital_map
in the info.json. For cases of spinful=true, the overlap.h5 remains
unchanged but the hamiltonian.h5 and density_matrix.h5 are expanded to
include the spin. The chunk_shapes doubles and the chunk_boundaries
becomes four times as before. Each block is composed by 4 parts as

                                          

                  Ai,j,R = Ai,j,R,↑,↑     Ai,j,R,↑,↓  (4.1)
                               Ai,j,R,↓,↑  Ai,j,R,↓,↓

                  with each part the same size as the non-sipnful case.
                  Note: The atom_pairs in all the *.h5 files inside the same directory should be the same.

                  4.2. USING DEEPH-DOCK TO PREPARE DFT DATA

                           Currently, the DeepH framework supports data from most mainstream DFT softwares,
                           including Quantum ESPRESSO, OpenMX, SIESTA, FHI-aims and ABACUS.

                           To provide users with a more streamlined and eﬀicient data generation solution, we
                           have developed the DeepH-dock project. This tool enables one-command generation
                           of configuration files for target computational software based on user-provided struc-
                           tures. Additionally, it automatically processes raw DFT output data into a standard-
                           ized format compatible with DeepH-pack (See Section 4.1), ensuring seamless integration
                           and high eﬀiciency in DFT-based workflows. For more details of the data interfaces
                           and other tools in DeepH-dock, please refer to the user guide of DeepH-dock (https:
                          //github.com/kYangLi/DeepH-dock.git).

DATA PREPARATION 4.2.1. Converting from Legacy Format

                           The legacy DFT data format adopted by DeepH and DeepH-E3 can be converted to the
                           current format by DeepH-dock:

                        1 $ dock convert deeph upgrade <legacy> <updated> -p PARALLEL_NUM

                           which converts all of the legacy DFT data from folder <legacy> to <updated> with
                           PARALLEL_NUM parallel processes. Please refer to the documentation of DeepH-dock for
                           more details.

                  4.2.2. FHI-aims

                           FHI-aims, one of the most successful all-electron localized orbital DFT software pack-
                           ages. We are excited to announce that DeepH-pack has now achieved deep integration
                           with FHI-aims. The integration methodology and interfaces will be unveiled for the first
                           time at the FHI-aims workshop in Shanghai, November 2025, and will subsequently be
                           incorporated into this manual.

                                                        9

4.2.3. Quantum ESPRESSO

                           The input of DeepH-pack is a DFT Hamiltonian under AO basis. To work with the plane-
                           wave DFT software e.g. Quantum ESPRESSO, we need to project the Hamiltonian onto an
                           AO basis (e.g., the orbitals generated in SIESTA). Please follow the example in HPRO,
                           which can generate both legacy and updated DeepH-pack DFT data.

                  4.2.4. OpenMX

                            OpenMX is a DFT calculation program based on norm-conserving pseudopotentials and
                            pseudo-atomic localized basis functions. To get the data required by DeepH-pack (overlap
                            matrix, Hamiltonian matrix, etc) from the output of OpenMX, one needs to set "HS.fileout
                            On" in the input *.in file. After the calculation is normally finished, all of the raw output
                            data is stored in the *.scfout file. Then the raw data can be converted to the standard
                            data format of DeepH-pack by the following command:

                         1 $ dock convert openmx to-deeph <openmx_dir > <deeph_dir > -p PARALLEL_NUM

                  4.3. NOTES ON DFT CALCULATIONS

                  4.3.1. The quality of the basis set

                           AO basis sets are non-orthogonal, which may lead to numerical instabilities in the post-
                           process of DeepH. The band structure of an AO Hamiltonian is obtained from a generalized
                           eigenvalue problem:

                  H(k)|ψk⟩ = ε(k)S(k)|ψk⟩                (4.2)
                                                         (4.3)
                  ∑                                   ∑
                  |ψk⟩ = ciα(k) eikR|ϕiα(R)⟩

                                                  iα  R

DATA PREPARATION If the condition number of S(k) is very large, a small
prediction error in H(k) can result in a large error in the predicted
band structure ε(k), and may result in "imaginary bands" near the Fermi
level of the predicted band structure. Therefore, working with a "good

                  basis” can make the best use of DeepH’s prediction power. Empirically, if the condition
                  number of the overlap matrix is less than 104 for a system with less than 100 atoms, the
                  basis can be considered as well-behaved.

                                                         10

DATA PREPARATION 5. BUILDING THE DATA GRAPH

                  5.1. WHAT IS THE GRAPH FILE

                         DeepH models are graph neural networks (GNNs). They take atomic structures as input
                         and predict physical quantities. The input structures are treated as graphs with atoms
                         as nodes. Any pair of atoms i and j are connected by directed edges i → j and j → i if
                         they’re suﬀiciently close (i.e., their atomic orbital basis functions overlap). There are also
                         self-loops i → i in the graph. Physical quantities, such as Hamiltonian matrix elements,
                         are interpreted as “features” associated with the nodes and edges of the graph.

                         Technically, graph files are directly constructed from DFT data, demonstrating complete
                         data equivalence with the DeepH training process. Compared to traditional folder-based
                         decentralized DFT data storage methods, the graph file system exhibits multiple technical
                         advantages:

                         • Numerical Precision Flexibility: DeepH-pack supports user-defined 32-bit or 64-
                             bit floating point precision storage, significantly enhancing storage eﬀiciency through
                             optimized data type configurations.

                         • Unified Data Portability: Leveraging a single-file integrated architecture, graph files
                             should be prioritized over raw fragmented data during cross-server cluster transfers to
                             streamline data mobility.

                         • Generalized Compatibility: Designed with a universal data structure, the graph
                             file format is not only compatible with the DeepH framework but also theoretically
                             extensible to training workflows of diverse neural network architectures.

                         The root directory for the graph files must be named as graph/, with all graph files
                         residing within this directory. DeepH-pack currently supports two distinct storage modes
                         for graph files: memory mode and disk mode. The former pre-loads the entire graph file
                         into node memory during DeepH training initialization, prioritizing operational eﬀiciency
                         for datasets compatible with available memory resources. The memory mode graph layout
                         looks like this:

                           graph
                               |- <GRAPH_NAME >.<GRAPH_TYPE >.memory.pt

                         The latter employs on-demand data streaming through integrated database-hardware stor-
                         age solutions, specifically designed for over-sized graph files exceeding node memory ca-
                         pacity (e.g., >10 TB). The disk mode graph layout looks like this:

                           graph
                               |- <GRAPH_NAME >.<GRAPH_TYPE >.disk.pt
                               |- <GRAPH_NAME >.<GRAPH_TYPE >.disk.part1 -of-1.db/
                               |- <GRAPH_NAME >.<GRAPH_TYPE >.disk.part1 -of-1.info.pt

                         This dual-mode architecture ensures memory-agnostic training workflows by dynamically
                         adapting to data scales, where disk mode enables real-time access during computation
                         while bypassing full memory occupancy, thereby maintaining system flexibility across
                         varying computational constraints.

                         For comprehensive documentation regarding the <GRAPH_NAME> and <GRAPH_TYPE> param-
                         eters, see Section 7.1 for implementation guidelines and technical specifications.

                                                                                                                                             11

DATA PREPARATION 5.2. BUILDING THE GRAPH

                           Upon initiating a standard DeepH training session, the framework automatically con-
                           structs graph files from DFT data stored in the designated "dft/" directory and gen-
                           erates the corresponding graph dataloader. Given the CPU-exclusive nature of graph
                           construction and the inherent advantages of graph files in data portability, DeepH-pack
                           also supports decoupled graph generation from the GPU-accelerated training process. If
                           graph files already exist, the training sessions would skip raw DFT data, streamlining the
                           training workflow through graph-based data abstraction. Furthermore, as demonstrated in
                           Section 6.1, pre-constructed graphs are a prerequisite for systematic data characterization
                           and subsequent training parameter optimization.
                           build_graph.toml:

                             # ----------------------------- SYSTEM -----------------------------
                             [system]
                             note = "Welcome to DeepH-pack!"
                             device = "cpu"
                             float_type = "fp32" # or `fp64`
                             random_seed = 137
                             # ----------------------------- DATA -------------------------------
                             [data]
                             inputs_dir = "." # Inputs path that contains `dft` and `graph `
                             outputs_dir = "./build_graph_logs" # Logging path
                             [data.dft]
                             data_dir_depth = 0 # dft data depth
                             [data.graph]
                             dataset_name = "SACADA -427"
                             graph_type = "HS" # See the Doc. for more detailed options
                             storage_type = "memory" # or `disk`
                             common_orbital_types = "" # See the Doc. for more detailed info.
                             parallel_num = -1 # Parallel processes during build graph
                             only_save_graph = true # A task for generate and save graph only

                          You can use this command to build data graph file or you can directly kick-off the training
                           process(see next chapter) and graph file will be built automatically in pre-train stage.

                        1 $ deeph-train build_graph.toml

                                                                                                                                              12

6. DATA ANALYSIS

                  6.1. ANALYSIS OF DATASET FEATURES

                  Prior to executing training tasks, performing comprehensive dataset feature analysis is
                  essential. This process enables systematic evaluation of training memory requirements
                  through statistical characteristics (e.g., edge distribution in individual graph structures),
                  informs parameter configuration strategies (including neural network irreducible repre-
                  sentations), and guides performance expectation assessments while establishing model
                  evaluation baselines.

                  To perform DFT dataset feature analysis, you must first configure a structured input
                  directory adhering to the specified schema:

                  data_inputs
                     |- dft
                         |- 0
                             |- POSCAR
                             |- info.json
                             |- overlap.h5
                             |- hamiltonian.h5 (optional)
                             |- density_matrix.h5 (optional)
                         |- 1
                         |- ...
                     |- graph
                         |- <GRAPH_NAME >.<GRAPH_TYPE >.memory.pt
                         |- <GRAPH_NAME >.<GRAPH_TYPE >.disk.pt
                         |- <GRAPH_NAME >.<GRAPH_TYPE >.disk.part1 -of-1.db/
                         |- <GRAPH_NAME >.<GRAPH_TYPE >.disk.part1 -of-1.info.pt

DATA PREPARATION whose path should be explicitly defined under the
\[data.inputs\] key in your TOML configuration file (the file which we
will introduce detailed latter). During training work- flows, the
presence of either DFT raw data or preprocessed graph files within this
directory suﬀices to initiate model training. However, comprehensive
dataset characterization re- quires concurrent availability of both data
modalities, as the mutually consistent DFT and graph subdirectories
collectively form the foundational training dataset. The analysis
process comprises two sequential phases:

                  • DFT metadata inspection via:

                  1 $ deeph-Tool-InspectDataset ./data_inputs -k 'dft'

                  2 ---------------------------------------------------------------

                  3 [info] Spinful:               False

                  4 [info] User needs parity:     False

                  5 [info] DFT data quantity:     427

                  6 [info] Elements included:     ['C']

                  7 [info] Common orbital types: s2p2d1

                  8 [info] Irreps common orbital: 9x0e+17x1e+13x2e+5x3e+1x4e (169, regrouped)

                  9 [info] Irreps in as suggested: 16x0e+24x1e+16x2e+8x3e+2x4e (242)

                  10 [info] Irreps in as exp2:    16x0e+8x1e+4x2e+2x3e+2x4e (92)

                  11 [info] Irreps in as trivial : 16 x0e +16 x1e +16 x2e +16 x3e +16 x4e (400)

                  12 ---------------------------------------------------------------

                  • Graph validation using:

                  1 $ deeph-Tool-InspectDataset ./data_inputs -k 'graph' --graph_pt_name <
                             GRAPH_NAME >.<GRAPH_TYPE >.disk.pt

                  2 ---------------------------------------------------------------

                                                                                                 13

3 \[info\] Dataset Name: SACADA -427

4 \[info\] Graph Type: train -HS

5 \[info\] Elements Included: \['C'\]

6 \[info\] Spinful: False

7 \[info\] Common Orbital Types: s2p2d1

8 \[info\] Common Fitting Types: None

9 \[info\] Structure Quantity: 427

10 \[info\] All Entries : 252704517

11 \[info\] Shape Masked in All: 100.0%

12 \[info\] Real Masked in Shape : 96.9%

13 \[info\] Real Entries : 244819714

14 ---------------------------------------------------------------

For detailed parameter specifications and operational guidance, execute
the following com- mand to access the integrated help documentation.

1 deeph-Tool-InspectDataset -h

                                                                    14

TRAINING MODELS

                                                                                                                                         15

TRAINING MODELS 7. USER CONFIGURATION

                 7.1. TRAINING

                        Parameters of training process are configured through a TOML-formatted file, where each
                        key systematically governs specific aspects of the computational workflow. An example
                        TOML is shown in below:

                        train.toml:

                          # ---------------------------------- SYSTEM ----------------------------------
                          [system]
                          note = "Enjoy DeepH -pack! ;-)"
                          device = "gpu*8:0"
                          float_type = "fp32"
                          random_seed = 137
                          log_level = "info"
                          jax_memory_preallocate = true
                          show_train_process_bar = true

                          # ----------------------------------- DATA ------------------------------------
                          [data]
                          inputs_dir = "./user/should/set/this/inputs"
                          outputs_dir = "./user/should/set/this/outputs"

                          [data.dft]
                          data_dir_depth = 0

                          [data.graph]
                          dataset_name = "DATASET-DEMO"
                          graph_type = "H"
                          storage_type = "memory"
                          common_orbital_types = ""
                          parallel_num = -1
                          only_save_graph = false

                          [data.model_save]
                          best = true
                          latest = true
                          latest_interval = 100
                          latest_num = 10

                          # ----------------------------- MODEL -----------------------------------------
                          [model]
                          net_type = "normal"
                          target_type = "H"
                          loss_type = "mse"

                          [model.advanced]
                          gaussian_basis_rmax = 7.5
                          net_irreps = ""
                          num_blocks = 3
                          consider_parity = true
                          standardize_gauge = false

                          # ------------------------------ PROCESS --------------------------------------
                          [process.train]
                          max_epoch = 10000

                          multi_way_jit_num = 1
                          ahead_of_time_compile = false

                                                                                                                                           16

TRAINING MODELS \[process.train.dataloader\] batch_size = 1

                  train_size = 1
                  validate_size = 0
                  test_size = 0
                  dataset_split_json = ""
                  only_use_train_loss = false

                 [process.train.optimizer]
                  type = "adamw"
                  init_learning_rate = 2E-3
                  clip_norm_factor = -1.0
                 # sgd
                  momentum = 0.8
                 # adam(w)
                  betas = [0.9, 0.999]
                  weight = 0.001
                  eps = 1E-8

                 [process.train.scheduler]
                  min_learning_rate_scale = 1E-4
                  type = "reduce_on_plateau"
                 # Reduce on plateau
                  factor = 0.5
                  patience = 500
                  rtol = 0.05
                  cooldown = 100
                  accum_size = -1
                 # Warmup cosine decay
                  init_scale = 0.1
                  warmup_steps = 10
                  decay_steps = -1
                  end_scale = -1.0

                 [process.train.continued]
                  enable = false
                  new_training_data = false
                  new_optimizer = false
                  previous_output_dir = ""
                  load_model_type = "latest"
                  load_model_epoch = -1

                 Next, we will go through the semantics of these parameters of the TOML file in detail.

                 System

                 [system.note]

                 • Description: Name of this training project.
                 • Default: "Enjoy DeepH-pack! ;-)"
                 • Type: <STRING>

                 [system.device]

                 • Description: The device configuration follows the syntax <type>*<num>:<id>, where

                                                                                                                                    17

TRAINING MODELS `<type>`{=html} specifies hardware type (cpu, gpu, rocm,
dcu, or cuda), `<num>`{=html} denotes either the total devices per node
(for accelerators like GPU) or the number of CPU partitions (when using
cpu), and `<id>`{=html} defines target device indices (e.g., gpu*8:1-4,7
selects 5 GPUs from an 8-device node using indices 1,2,3,4,7). Note: For
CPU configurations, `<id>`{=html} is ignored while `<num>`{=html}
controls thread partitioning. • Default: "gpu*8:0" • Type:
`<STRING>`{=html}

                 [system.float_type]
                 • Description: The float type for building graph and training model. For most of the

                    Hamiltonian tasks, "fp32" is accurate enough with float point error around 0.001 ∼ 0.01
                    meV.
                 • Default: "fp32"
                 • Type: ["bf16", "tf32", "fp32", "fp64"]

                 [system.random_seed]
                 • Description: The seed for generating random numbers.
                 • Default: 137
                 • Type: <INT>

                 [system.log_level]
                 • Description: The degree of severity for deepx.log, increasing in the order of debug,

                    info, warning, error, and critical.
                 • Default: "info"
                 • Type: ["debug", "info", "warning", "critical"]

                 [system.jax_memory_preallocate]
                 • Description: Whether to pre-allocate 75% of the remaining memory before running.

                    Use true for formal training; use false for debugging to observe memory usage on
                    the GPU. Using false during training may cause the process to unexpectedly crash
                    mid-way or slow down the training process.
                 • Default: true
                 • Type: <BOOL>

                 [system.show_train_process_bar]
                 • Description: Whether to display a graphical progress bar in the command line.
                 • Default: true
                 • Type: <BOOL>

                                                                                                                                    18

TRAINING MODELS Data

                 [data.inputs_dir]

                 • Description: Specify the root directory containing structured input data with required
                    sub-folders dft/ or graph/ (see Section 6.1). Accepts both relative paths (resolved from
                    execution context) and absolute paths (system-native format).

                 • Default: <Invalid-Input>
                 • Type: <STRING>

                 [data.outputs_dir]

                 • Description: The output directory designates the parent location for storing training
                    artifacts (log files and serialized models). By default, the system automatically gen-
                    erates timestamped subdirectories (ISO 8601 extended format: %Y-%m-%d_%H-%M-%S)
                    within user-specified paths, isolating each training session’s outputs. Full structural
                    specifications detailed in Section 9.1 (Post-Training Workflows).

                 • Default: <Invalid-Input>
                 • Type: <STRING>

                 [data.dft.data_dir_depth]

                 • Description: When organizing DFT training data subdirectories, if the number of
                    structural configurations becomes excessive (e.g., reaching 100,000 structures), a flat
                    directory structure under inputs_dir/dft/* becomes impractical. In such cases, a
                    hierarchical folder architecture is recommended, such as dft/<t1>/<t2>/*. This con-
                    figuration establishes a directory depth of 2, data_dir_depth=2, providing a scalable
                    storage solution for massive structural datasets while maintaining systematic accessibil-
                    ity.

                 • Default: 0
                 • Type: <INT>

                 [data.graph.dataset_name]

                 • Description: Name of your dataset.
                 • Default: "DATASET-DEMO"
                 • Type: <STRING>

                 [data.graph.graph_type]

                 • Description: The physical quantity used for training. DeepH will build a correspond-
                    ing graph file. One can choose from H, S, Rho, HS, and Sap. The H for Hamiltonian,
                    Rho for density matrix, S for overlap, HS for both Hamiltonian and overlap, and Sap for
                    shape-only overlap. The vertices of the graph are atoms, and there is an edge between
                    two atoms if they are close.

                 • Default: "H"

                                                                                                                                    19

TRAINING MODELS • Type: \["H", "HS", "Rho", "Sap", "S"\]

                 [data.graph.storage_type]
                 • Description: Where to store the graph data. Choose "memory" to store in memory.

                    For very large datasets, choose "disk" to store on disk.
                 • Default: "memory"
                 • Type: ["memory", "disk"]

                 [data.graph.common_orbital_types]
                 • Description: The list of one atom’s orbitals arranged according to the value of l(angular

                    quantum number). e.g., s2p2d1. Default (set to "") is the union of orbital types of all
                    different atoms in the dataset. For example, for basis sets Mo-s3p2d1 and S-s2p2d1
                    in the OpenMX calculation￿the union of orbital types is [0, 0, 0, 1, 1, 2], which
                    corresponds to s3p2d1.
                 • Default: ""
                 • Type: <STRING>

                 [data.graph.parallel_num]
                 • Description: Determines the maximum concurrent parallel processes allocated for

                    graph construction. When configured with non-positive integers or values exceeding the
                    available compute resources (i.e., surpassing either the host’s physical CPU core count
                    or accelerator device quantity), the system will dynamically scale the parallelism based
                    on hardware availability - specifically adopting the greater value between detected CPU
                    cores and accelerator devices (GPUs) to optimize computational throughput.
                 • Default: -1
                 • Type: <INT>

                 [data.graph.only_save_graph]
                 • Description: If set to true, the program will only generate and save the graph file to

                    file-system and quit.
                 • Default: false
                 • Type: <BOOL>

                 [data.model_save.best]
                 • Description: Whether to save the model with the lowest loss.
                 • Default: true
                 • Type: <BOOL>

                                                                                                                                    20

TRAINING MODELS \[data.model_save.latest\]

                 • Description: Whether to save the model in the latest training epoch.
                 • Default: true
                 • Type: <BOOL>

                 [data.model_save.latest_interval]

                 • Description: Only functional when data.model_save.latest is true. This param-
                    eter governs the checkpointing frequency for model state preservation during training.
                    When configured with non-positive values or exceeding the maximum epoch count, the
                    system enforces periodic snapshots at epoch multiples of the specified integer - e.g., set-
                    ting latest_interval=10 systematically generates training state archives at epochs 10,
                    20, 30, etc., implementing epoch-aligned preservation of complete model states (weights,
                    optimizer parameters, and metadata).

                 • Default: 100
                 • Type: <INT>

                 [data.model_save.latest_num]

                 • Description: Only functional when data.model_save.latest is true. The number
                    of latest checkpoints that user wants to keep.

                 • Default: 10
                 • Type: <INT>

                 Model

                 [model.net_type]

                 • Description: The neural network architecture.
                    – sparrow (also named normal) is a light-weighted architecture (typically <1M param-
                        eters) with both node and edge features, which is suitable for small tasks of DFT
                        Hamiltonian learning.
                    – eagle (also named accurate) is an advanced architecture (typically ∼5M parameters)
                       with both node and edge features, which is suitable for tasks of DFT Hamiltonian
                        learning that requires high accuracy.

                 • Default: "normal"
                 • Type: ["sparrow", "normal", "eagle", "accurate"]

                 [model.target_type]

                 • Description: The physical quantity to learn. H for Hamiltonian, and Rho for density
                    matrix.

                 • Default: "H"

                                                                                                                                    21

TRAINING MODELS • Type: \["H", "Rho""\]

                 [model.loss_type]

                 • Description: Loss function during training. The loss type not only support the super-
                    vised ones (mse, mae, etc.) but also support on-the-fly unsupervised loss (like ai2dft,
                    ai2dft_node, hopad, and aims).

                 • Default: "mse"
                 • Type: ["mae", "mse", "wmae", "huber", "ai2dft", "ai2dft_node", "hopad",

                    "aims"]

                 Model: Advanced

                 [model.advanced.gaussian_basis_rmax]

                 • Description: The cutoff radius used for Gaussian basis sampling, in angstrom. We
                    suggest set this cutoff to 2×the maximum cutoff radius of your orbital basis. Refer the
                    paper of DeepH-E3 for details.

                 • Default: 7.5
                 • Type: <FLOAT>

                 [model.advanced.net_irreps]

                 • Description: Irreducible representations of the neural network features, which ensure
                    the equivariance of the network.
                    For sparrow, the Irreps can be set to “· · · x0e+· · · x1o+· · · x2e+· · · x3o+· · · x4e+· · · ”.
                    The channel l has parity (−1)l.
                    For eagle and owl, set to “· · · x0e+· · · x1e+· · · x2e+· · · x3e+· · · x4e+· · · ”. All the chan-
                    nels must have even parity.
                    Note: “64x1o” means that the feature is a 64-channel tensor, where each channel has
                    odd (“o”) parity and carries the l = 1 representation of the SO(3) group.
                    o/e refers to odd/even parity.
                    Set in the form of e3nn.Irreps, namely “irreducible representations”, which describes
                    the symmetry of input features. e.g. in the form of [(mul_l, (l, p_val · (p_arg)l))
                    for l ∈ [0, . . . , lmax]].

                 • Default: <Invalid-Input>
                 • Type: <STRING>

                 [model.advanced.num_blocks]

                 • Description: The number of neural network layers in the model. For usual task, we
                    recommend set 3 for solids, 4 for small molecules.

                 • Default: 3
                 • Type: <INT>

                                                                                                                                    22

TRAINING MODELS \[model.advanced.consider_parity\]

                 • Description: Whether the network is equivariant under parity. If set to false, the
                    net_irreps cannot appear odd (e.g., 2x3o) representations. In eagle and owl net, the
                    consider_parity must be false.

                 • Default: true
                 • Type: <BOOL>

                 [model.advanced.standardize_gauge]

                 • Description: Whether to consider arbitrariness of gauge (zero-energy point in DFT)
                    when training the model. If set to “true”, graph_type must be HS. We suggest set
                    true for a general-purpose dataset (or whenever the training data has a zero-energy
                    arbitrariness problem, e.g., a dataset with 2D slabs of different thicknesses), false for
                    a special-purpose dataset.

                 • Default: false
                 • Type: <BOOL>

                 Process: Train

                 [process.train.max_epoch]

                 • Description: The maximum number of epochs. Training will automatically stop when
                    epoch_number reaches max_epoch.

                 • Default: 10000
                 • Type: <INT>

                 [process.train.multi_way_jit_num]

                 • Description: This helps accelerate the training process when the number of edges
                    between different structures in the dataset varies greatly. Recommended value is 10-20.
                    This may cause the first epoch of training to be slow and may cause out-of-memory
                    error.

                 • Default: 1
                 • Type: <INT>

                 [process.train.ahead_of_time_compile]

                 • Description: Whether to use ahead-of-time (AOT) compilation to accelerate the JIT
                    (or more precisely, compile) process. With the AOT method help, the multi-way JIT
                    can speed up from 2 hours to 10 minutes.

                 • Default: false
                 • Type: <BOOL>

                                                                                                                                    23

TRAINING MODELS Process: Train: Dataloader

                 [process.train.dataloader.batch_size]
                 • Description: Batch size, number of structures in a batch.
                 • Default: 1
                 • Type: <INT>

                 [process.train.dataloader.train_size]
                 • Description: Number of structures in the training dataset.
                 • Default: 1
                 • Type: <INT>

                 [process.train.dataloader.validate_size]
                 • Description: Number of structures in the validation dataset.
                 • Default: 0
                 • Type: <INT>

                 [process.train.dataloader.test_size]
                 • Description: Number of structures in the test dataset.
                 • Default: 0
                 • Type: <INT>

                 [process.train.dataloader.dataset_split_json]
                 • Description: A JSON file path to execute dataset partitioning with customized rules.

                    To employ the framework’s default splitting mechanism, set this parameter to an empty
                    string (""). The JSON configuration must adhere to the following schema specification:

                     {"train": ["1", "3", "5"], "validate": ["2"], "test":["4","6"]}

                 • Default: ""
                 • Type: <STRING>

                 [process.train.dataloader.only_use_train_loss]
                 • Description: Whether to adjust the learning rate based only on the train loss rather

                    than considering both train and validation loss. If the validation set is empty (although

                                                                                                                                    24

TRAINING MODELS this is not common), the validation loss is always a
non-sense value, so it should be set to consider only train loss. •
Default: false • Type: `<BOOL>`{=html}

                 Process: Train: Optimizer

                 [process.train.optimizer.type]
                 • Description: The optimizer core type, can choose from: adamw, adam, or sgd.
                 • Default: "adamw"
                 • Type: ["sgd", "adam", "adamw"]

                 [process.train.optimizer.init_learning_rate]
                 • Description: Initial learning rate. We suggest use 2E-3 for net_type=sparrow, 1E-3

                    for net_type=eagle or owl. Larger learning rate can speed up convergence while may
                    be confronted with instabilities.
                 • Default: 2E-3
                 • Type: <FLOAT>

                 [process.train.optimizer.clip_norm_factor]
                 • Description: Enable the clip normalization algorithm in the optimizer to prevent

                    large gradients that can cause neuron death. This feature will be disabled, if it was set
                    with negative value.
                 • Default: -1.0
                 • Type: <FLOAT>

                 [process.train.optimizer.momentum]
                 • Description: For sgd optimizer only. Controls the influence of previous gradients on

                    the current gradient update.
                 • Default: 0.8
                 • Type: <FLOAT>

                 [process.train.optimizer.betas]
                 • Description: For adam and adamw optimizers. betas = [beta1, beta2], where beta1

                    controls the exponential decay rate of first moment estimate (i.e., momentum), and
                    beta2 controls the exponential decay rate of second moment estimate (i.e., the uncen-
                    tered variance) usually used for calculating the moving average of squared gradients.

                                                                                                                                    25

TRAINING MODELS • Default: \[0.9, 0.999\] • Type:
`<LIST-OF-FLOAT>`{=html}

                 [process.train.optimizer.eps]
                 • Description: For adam and adamw optimizers. A small constant to improve numerical

                    stability. In some cases it is useful to decrease it to 1E-10 or lower.
                 • Default: 1E-8
                 • Type: <FLOAT>

                 [process.train.optimizer.weight]
                 • Description: For adamw optimizer only. The learning weight decay rate in adamw, to

                    avoid overfitting.
                 • Default: 0.001
                 • Type: <FLOAT>

                 Process: Train: Scheduler

                 Within the optax optimization framework, learning rate decay operates through a decou-
                 pled control mechanism governed by the scale parameter. The effective learning rate for
                 neural network updates is determined by the product: learning_rate (lr) = init_lr ×
                 scale. A dedicated scheduler module systematically modulates this scaling factor through-
                 out training, enabling implementation of various decay strategies (step-wise, exponential,
                 or cosine decay) while maintaining architectural isolation between initialization values and
                 decay dynamics.

                 [process.train.scheduler.min_learning_rate_scale]
                 • Description: The minimum learning rate scale. Training will automatically stop when

                    the learning rate scale reaches min_learning_rate_scale.
                 • Default: 1E-4
                 • Type: <FLOAT>

                 [process.train.scheduler.type]
                 • Description: One can choose from: “ReduceOnPlateau”, and “WarmupCosineDecay”.
                 • Default: "reduce_on_plateau"
                 • Type: ["reduce_on_plateau", "warmup_cosine_decay"]

                 Reduce LR On Plateau

                                                                                                                                    26

TRAINING MODELS \[process.train.scheduler.factor\]

                 • Description: For "reduce_on_plateau" only. Every time the learning rate adjustment
                    is triggered, learning rate scale will be updated to factor×scale.

                 • Default: 0.5
                 • Type: <FLOAT>

                 [process.train.scheduler.patience]

                 • Description: For "reduce_on_plateau" only. The number of evaluation steps to
                    wait before reducing the learning rate. This helps determine when there is no further
                    improvement in model performance. Be careful that by default, the number of patience
                    step is defined based on the train steps (each batch), rather than epoch (each train-set).
                    In most cases, we suggest to define this patience according to your own dataset scale,
                    computing resources, and expected time cost. Usually patience=120 epochs gives a
                    well-converged model, and patience=60 epochs or less gives a relatively quick result.

                 • Default: 500
                 • Type: <INT>

                 [process.train.scheduler.rtol]

                 • Description: For "reduce_on_plateau" only. Relative tolerance. A loss is considered
                    no longer improving if its relative improvement over the previous best validation loss is
                    less than rtol.

                 • Default: 0.05
                 • Type: <FLOAT>

                 [process.train.scheduler.cooldown]

                 • Description: For "reduce_on_plateau" only. The minimum number of evaluation
                    cycles between two learning rate adjustments. This prevents frequent changes to the
                    learning rate. Be careful that by default, the number of cooldown step is defined based
                    on the train steps (each batch), rather than epoch (each train-set). In most cases,
                    cooldown=20∼50 epochs is enough.

                 • Default: 100
                 • Type: <INT>

                 [process.train.scheduler.accum_size]

                 • Description: For "reduce_on_plateau" only. The patience and cooldown param-
                    eters operate on gradient update steps rather than epochs, with their baseline values
                    calculated as training batches per epoch × target epochs. For instance, given 100 batches
                    in each epoch:
                    – To monitor validation loss for 20 epochs without gradient accumulation (accum_size=1),
                        set patience=100×20=2000.
                    – With accum_size=100 (effectively creating macro-batches of 100×batch_size), each
                       ”step” becomes equivalent to 1 epoch, thus patience=20 suﬀices.

                                                                                                                                    27

TRAINING MODELS -- When patience=-1, the system auto-configures it as
total training batches to imple- ment full-epoch monitoring cycles.

                    Cooldown steps follow equivalent computational logic based on gradient update granu-
                    larity.
                 • Default: -1
                 • Type: <INT>

                 Warmup Cosine Decay
                 By combining warmup and cosine decay, the scheduler helps deep learning models converge
                 faster and improves their performance.

                 [process.train.scheduler.init_scale]
                 • Description: For "warmup_cosine_decay" only. The initial scaling factor for learning

                    rate scale.
                 • Default: 0.1
                 • Type: <FLOAT>

                 [process.train.scheduler.warmup_steps]
                 • Description: For "warmup_cosine_decay" only. The number of steps for the learning

                    rate scale to linearly increase from init_scale to 1.0, just like ”warmup” a model.
                    This helps stabilize the model’s behavior during the early stages of training.
                 • Default: 1000
                 • Type: <INT>

                 [process.train.scheduler.decay_steps]
                 • Description: For "warmup_cosine_decay" only. The number of steps for the learning

                    rate scale to decay from 1.0 to end_scale. Note that training will stop early when the
                    number of epochs reaches the max_epoch.
                 • Default: 2E5
                 • Type: <INT>

                 [process.train.scheduler.end_scale]
                 • Description: For "warmup_cosine_decay" only. The scaling factor at the end of the

                    learning rate scheduling. Set to -1.0 for the learning rate to decay to 0. Note that train-
                    ing will stop early when the learning rate scale is lower than the min_learning_rate_scale.
                 • Default: -1.0
                 • Type: <FLOAT>

                                                                                                                                    28

TRAINING MODELS Process: Train: Continued

                 This section is for fine-tuning or continuing training on an existing model.

                 [process.train.continued.enable]
                 • Description: Set to true for continued training or fine-tuning from an existing model,

                    or false for starting from scratch.
                 • Default: false
                 • Type: <BOOL>

                 [process.train.continued.new_training_data]
                 • Description: Whether to use new training data. Set to true for fine-tuning like task,

                    and false when running on the same dataset as the previous one.
                 • Default: false
                 • Type: <BOOL>

                 [process.train.continued.new_optimizer]
                 • Description: Whether to use a new optimizer. Set to true for fine-tuning like task,

                    and false when running on the same optimizer and scheduler as the previous one.
                 • Default: false
                 • Type: <BOOL>

                 [process.train.continued.previous_output_dir]
                 • Description: Previous output directory with time stamp, which contains the deepx.log

                    file and the model folder.
                 • Default: <Invalid-Input>
                 • Type: <STRING>

                 [process.train.continued.load_model_type]
                 • Description: Continue training from the best or latest model as needed.
                 • Default: "latest"
                 • Type: ["best", "latest"]

                 [process.train.continued.load_model_epoch]
                 • Description: For load_model_type = "latest" only. Specify a number for particular

                    epoch exist saved in latest model folder. Use -1 for the most latest epoch.
                 • Default: "-1"

                                                                                                                                    29

TRAINING MODELS • Type: `<INT>`{=html}

                 7.2. INFERENCE

                         Inference process configurations are defined through a TOML-formatted file, where each
                         key systematically governs specific aspects of the computational workflow. The hierar-
                         chical structure and semantic definitions of these configuration parameters are explicitly
                         detailed in subsequent sections.

                         infer.toml:

                           # ----------------------------- SYSTEM -----------------------------
                           [system]
                           note = "Enjoy DeepH -pack! ;-)"
                           device = "gpu*8:0"
                           float_type = "fp32"
                           random_seed = 137
                           log_level = "info"
                           jax_memory_preallocate = true

                           # ------------------------------ DATA -------------------------------
                           [data]
                           inputs_dir = "./user/should/set/this/inputs"
                           outputs_dir = "./user/should/set/this/outputs"

                           [data.dft]
                           data_dir_depth = 0

                           [data.graph]
                           dataset_name = "INFER-DEMO"
                           graph_type = "S"
                           storage_type = "memory"
                           parallel_num = -1
                           only_save_graph = false

                           # ----------------------------- MODEL -------------------------------
                           [model]
                           model_dir = "./user/should/set/this"
                           load_model_type = "best"
                           load_model_epoch = -1

                           # ---------------------------- PROCESS ------------------------------
                           [process.infer]
                           output_type = "h5"
                           output_into = "to_output"
                           target_symmetrize = true
                           multi_way_jit_num = 1

                           [process.infer.dataloader]
                           batch_size = 1

                                                                                                                                            30

TRAINING MODELS System

                 [system.note] The same as training.
                 [system.device] The same as training.
                 [system.float_type] The same as training.
                 [system.random_seed] The same as training.
                 [system.log_level] The same as training.
                 [system.jax_memory_preallocate] The same as training.

                 Data

                 [data.inputs_dir] The same as training.
                 [data.outputs_dir] The same as training.
                 [data.dft.data_dir_depth] The same as training.
                 [data.graph.dataset_name] The same as training.
                 [data.graph.graph_type]
                 • Description: The physical quantities needed for inference. DeepH will build a corre-

                    sponding graph file. One can choose from Sap and S. The S for overlap, Sap for overlap
                    but do not calculate mask using overlap values.
                 • Default: "S"
                 • Type: ["Sap", "S"]

                 [data.graph.storage_type] The same as training.
                 [data.graph.parallel_num] The same as training.
                 [data.graph.only_save_graph] The same as training.

                                                                                                                                    31

TRAINING MODELS Model

                 [model.model_dir]
                 • Description: The directory storing the trained model, usually with the format of

                    <time_stamp>/model.
                 • Default: <Invalid-Input>
                 • Type: <STRING>

                 [model.load_model_type]
                 • Description: Infer with best or latest trained model.
                 • Default: "best"
                 • Type: ["best", "latest"]

                 [model.load_model_type]
                 • Description: For load_model_type = "latest" only. Specify a number for particular

                    epoch aved in latest model folder. Use -1 for the most latest epoch.
                 • Default: -1
                 • Type: <INT>

                 [model.process.infer.output_type]
                 • Description: The output format.
                 • Default: "h5"
                 • Type: ["h5", "petsc"]

                 [model.process.infer.output_into]
                 • Description: Location for storing the predicted data. One can choose from a new folder

                    under output path (<time_stamp>/dft) or the original data folder (<inputs>/dft).
                    The output Hamiltonians are named as hamiltonian_pred.h5.
                 • Default: "to_output"
                 • Type: ["to_output", "to_input"]

                 [model.process.infer.target_symmetrize]
                 • Description: Whether to symmetrize the predicted target (e.g., to hermitianize the

                    Hamiltonian).

                                                                                                                                    32

TRAINING MODELS • Default: true • Type: `<BOOL>`{=html}
\[model.process.infer.multiway_jit_num\] The same as training
model.process.train.multiway_jit_num.
\[model.process.infer.dataloader.batch_size\] • Description: Batch size
for inference. Can be significantly larger than the training

                    batch size.
                 • Default: 1
                 • Type: <INT>

                                                                                                                                    33

TRAINING MODELS 8. TRAINING AND INFERENCE

                 8.1. RUN DEEPH

                        As mentioned before, in order to run DeepH (either model training or inference), an input
                         folder (in specific layout structure) is needed along with a number of input files (material
                         data and/or graph data):

                           data_inputs
                              |- dft
                                  |- 0
                                     |- POSCAR
                                     |- info.json
                                     |- overlap.h5
                                     |- hamiltonian.h5 (optional)
                                     |- density_matrix.h5 (optional)
                                  |- 1
                                  |- ...
                              |- graph
                                  |- <GRAPH_NAME >.<GRAPH_TYPE >.memory.pt
                                  |- <GRAPH_NAME >.<GRAPH_TYPE >.disk.pt
                                  |- <GRAPH_NAME >.<GRAPH_TYPE >.disk.part1 -of-1.db/
                                  |- <GRAPH_NAME >.<GRAPH_TYPE >.disk.part1 -of-1.info.pt

                        And a train.toml or infer.toml needs to be configured, then you are good to go.
                         Use the following command to start a training process:

                      1 $ deeph-train train.toml

                         Use the following command to start an inference process:

                      1 $ deeph-infer infer.toml

                         Use the following command to monitor the training. The train error, validation error, and
                         learning rate scale of the training process will be plotted:

                      1 $ deeph -Tool-PlotTrainingProcess <output_dir >

                                                                                                                                            34

TRAINING MODELS 9. AFTER TRAINING AND INFERENCE

                 9.1. AFTER TRAINING

                          Once training is done, you get an output directory like this:

                            outputs/<TIME_STAMP >
                               |- dataset_split.json
                               |- deepx.log
                               |- model
                                       |- train.toml
                                       |- variables.json
                                       |- params
                                          |- best.pytree
                                                  |- epoch_124/
                                          |- latest.pytree
                                                  |- epoch_120/
                                                  |- epoch_110/
                                                  |- epoch_100/
                                                  |- ...
                                       |- states
                                          |- best.pytree
                                          |- latest.pytree

                          Use the following command to analyze the model parameters:

                       1 $ deeph-Tool-AnalysisModelParams -h

                 9.2. DATA POST-PROCESSING

                          DeepH-dock is all you need! Please refer to the documentation of DeepH-dock (https:
                         //github.com/kYangLi/DeepH-dock.git).

                                                                                                                                             35


