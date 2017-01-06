# TFF n-d array
General-purpose _n_-dimensional array for C++, with many features useful for handling dense data.
Developed as part of _timed flow framework_ (TFF), but independent.
Encapsulates low-level issues such as indexing and alignment in dealing with raw data, allowing for readable and
efficient application code. Extensively tested.

Some features:

* `ndarray<Dim, T>` type inspired by _numpy_'s ndarray. C++ `Container` with random-access iterators. Sectioning
  and slicing giving non-owning read-only or read-write `ndarray_view<Dim, T>` with same interface. Byte-level strided
  data. Axis can be reversed. Convenient slicing syntaxes. Optimized copying and comparing for POD element types `T`.
  Row-major, column-major, or any other data ordering. *Timed* variant for each view, where absolute time index is
  associated to first dimension.

* **Intra-element slicing** for appropriate element types. For example `ndarray_view<2, std::array<int, 3>>` can be
  accessed as `ndarray_view<3, int>`. Standard-layout tuple type `elem_tuple`, allowing access to array of single
  component in packed multi-component _n_-d array.

* **Wrap-around view** of a smaller `ndarray_view`, where coordinate that cross the boundaries of the original view are
  mapped in a circular fashion. Same interface as normal views, including iteration and further sectioning.

* **Opaque view** where elements are replaced by frames or runtime-determined size and structure. Casting from and back
  to concrete `ndarray_view`, rendering given number of inner dimensions opaque. Provides type erasure and removes
  unnecessary templatization in application code operating on arbitrary data. For example `ndarray<3, int>` may be
  casted to `ndarray_opaque_view<1, ndarray_frame>`, and then back to `ndarray<3, int>`. Runtime verification of type.
  For POD-data (even strided), assignment and comparison possible in opaque state.

* **Opaque array** which is created and allocated in opaque state. Frames of may be of application-defined structure,
  instead of being casted `ndarray`s. Runtime-determined alignment requirement of frames is respected. Integrates
  application-defined *frame handle* class for convenient access to frame content.
