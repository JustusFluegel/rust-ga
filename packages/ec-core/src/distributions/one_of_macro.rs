/// Create a Distribution of a specified type, calling `.into()` on all
/// elements. Similar to the `[...]` syntax of rust optionally allowing to
/// specify the type using `uniform_distribution_of![<T> ...]`. To sample from
/// the Distribution, the elements are cloned. If you wish to avoid that, use
/// `uniform_distribution_of![ref <T> ...]` to return references instead.
///
/// This is equivalent to `arr_into![].into_distribution().unwrap()` without the
/// need for an `unwrap()`.
///
/// ![Railroad diagram for the `uniform_distribution_of!` macro][ref_text]
///
/// # Examples
/// ```rs
/// let distrb_1: impl Distribution<&i64> = uniform_distribution_of![ref <i64>
///     1i32,
///     2i32
/// ];
/// let distrb_2: impl Distribution<i64> = uniform_distribution_of![<i64>
///     1i32,
///     2i32
/// ];
/// ```
#[macro_railroad_annotation::generate_railroad("ref_text")]
#[macro_export]
macro_rules! uniform_distribution_of {
     (<$output_type:ty>  $($items:expr),+ $(,)?) => {
          unsafe {
               ::std::result::Result::unwrap_unchecked(
                    $crate::distributions::conversion
                         ::IntoDistribution::<$output_type>::into_distribution([
                         $(::std::convert::Into::<$output_type>::into($items)),+
                    ])
               )
          }
     };
     ($($items:expr),+ $(,)?) => {
          unsafe {
               ::std::result::Result::unwrap_unchecked(
                    $crate::distributions::conversion::IntoDistribution::into_distribution([
                         $($items),+
                    ])
               )
          }
     };
     (ref <$output_type:ty> $($items:expr),+ $(,)?) => {
          uniform_distribution_of![<$output_type>
               $(&$items)+
          ]
     };
     (ref  $($items:expr),+ $(,)?) => {
          uniform_distribution_of![
               $(&$items)+
          ]
     };
}

#[cfg(test)]
mod test {
    use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};

    use crate::distributions::conversion::{IntoDistribution, ToDistribution};

    #[test]
    fn distr_into_owned() {
        let distr = uniform_distribution_of![<i64>
             1i32,
             3i32
        ];

        assert_eq!(distr, [1i64, 3i64].into_distribution().unwrap())
    }

    #[test]
    fn distr_inferred_owned() {
        let distr = uniform_distribution_of![1i32, 3i32];

        assert_eq!(distr, [1i32, 3i32].into_distribution().unwrap())
    }

    #[test]
    fn distr_into_borrowed() {
        let distr_1 = uniform_distribution_of![ref <i64>
             1i32,
             3i32
        ];

        // rand::distribution::Slice<i64> doesn't impl eq so use this instead

        let distr_2 = ToDistribution::<&_>::to_distribution(&[1i64, 3i64]).unwrap();

        let seed: u64 = thread_rng().gen();
        let mut rng_1 = StdRng::seed_from_u64(seed);
        let mut rng_2 = StdRng::seed_from_u64(seed);

        for _ in 0..100 {
            let a = rng_1.sample(&distr_1);
            let b = rng_2.sample(distr_2);

            assert_eq!(&a, b);
        }
    }

    #[test]
    fn distr_inferred_borrowed() {
        let distr_1 = uniform_distribution_of![ref 1i32, 3i32];

        let distr_2 = ToDistribution::<&_>::to_distribution(&[1i32, 3i32]).unwrap();

        let seed: u64 = thread_rng().gen();
        let mut rng_1 = StdRng::seed_from_u64(seed);
        let mut rng_2 = StdRng::seed_from_u64(seed);

        for _ in 0..100 {
            let a = rng_1.sample(&distr_1);
            let b = rng_2.sample(distr_2);

            assert_eq!(&a, b);
        }
    }
}