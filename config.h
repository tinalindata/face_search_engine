//#define VERIFY 1
#ifndef DONKEY_XMATCH
#define DONKEY_XMATCH

#include <mutex>
#include <sstream>
#include <memory>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem/fstream.hpp>
#include "donkey-common.h"

namespace donkey {

    uint32_t constexpr DIM = 512;

    struct Feature: public VectorFeature<float, DIM> {
    };

    typedef distance::L2<float, DIM> FeatureSimilarity;

    // no data, but has weight
    struct Object: public SingleFeatureObject<Feature> {
    };

    typedef TrivialMatcher<Object, FeatureSimilarity> Matcher;

    class Extractor: public ExtractorBase {

    public:
        Extractor (Config const &config_) {
        }

        ~Extractor () {
        }

        void extract_url (string const &url, string const &type, Object *object) const {
        }

        void extract (string const &content, string const &type, Object *object) const {
        }
    };
}


#endif
