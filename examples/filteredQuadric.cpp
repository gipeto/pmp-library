// Copyright 2013-2017 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#include <pmp/SurfaceMesh.h>
#include <pmp/algorithms/SurfaceNormals.h>
#include <pmp/algorithms/DifferentialGeometry.h>
#include <Eigen/Core>

#include <nanoflann.hpp>

#include <array>
#include <iostream>

using namespace pmp;

namespace {

class Quadric
{
public: // clang-format off

    //! construct quadric from upper triangle of symmetrix 4x4 matrix
    Quadric(double a, double b, double c, double d,
            double e, double f, double g,
            double h, double i,
            double j)
        : a_(a), b_(b), c_(c), d_(d),
          e_(e), f_(f), g_(g),
          h_(h), i_(i),
          j_(j)
    {}

    //! constructor quadric from given plane equation: ax+by+cz+d=0
    Quadric(double a=0.0, double b=0.0, double c=0.0, double d=0.0)
        :  a_(a*a), b_(a*b), c_(a*c),  d_(a*d),
           e_(b*b), f_(b*c), g_(b*d),
           h_(c*c), i_(c*d),
           j_(d*d)
    {}

    //! construct from point and normal specifying a plane
    Quadric(const Normal& n, const Point& p)
    {
        *this = Quadric(n[0], n[1], n[2], -dot(n,p));
    }

    //! set all matrix entries to zero
    void clear() { a_ = b_ = c_ = d_ = e_ = f_ = g_ = h_ = i_ = j_ = 0.0; }

    //! add given quadric to this quadric
    Quadric& operator+=(const Quadric& q)
    {
        a_ += q.a_; b_ += q.b_; c_ += q.c_; d_ += q.d_;
        e_ += q.e_; f_ += q.f_; g_ += q.g_;
        h_ += q.h_; i_ += q.i_;
        j_ += q.j_;
        return *this;
    }

    //! multiply quadric by a scalar
    Quadric& operator*=(double s)
    {
        a_ *= s; b_ *= s; c_ *= s;  d_ *= s;
        e_ *= s; f_ *= s; g_ *= s;
        h_ *= s; i_ *= s;
        j_ *= s;
        return *this;
    }

    Quadric operator+(const Quadric& q) const
    {
        return Quadric(a_ + q.a_, b_ + q.b_, c_ + q.c_, d_ + q.d_,
        e_ + q.e_, f_ + q.f_, g_ + q.g_,
        h_ + q.h_, i_ + q.i_,
        j_ + q.j_);
        
    }

    //! multiply quadric by a scalar
    Quadric operator*(double s) const
    {
        return Quadric(a_ * s, b_ * s, c_ * s,  d_ * s,
        e_ * s, f_ * s, g_ * s,
        h_ * s, i_ * s,
        j_ * s);
    }

    Point solve() const
    {
         Eigen::Matrix<double,3,3> A;
          A << a_, b_, c_,
             b_, e_, f_, 
             c_, f_, h_;

        Eigen::Matrix<double,3,1> b{d_,g_,i_};

        Eigen::Matrix<double,3,1> p = -A.colPivHouseholderQr().solve(b);

        double relative_error = (A*(-p) - b).norm() / b.norm();
       
        if(relative_error > 1e-9)
        {
            std::cout << "Relative error: " << relative_error << std::endl;
            std::cout << "Detected singular matrix" << std::endl;
        }



        return Point(p.cast<float>());

    }

    //! evaluate quadric Q at position p by computing (p^T * Q * p)
    double operator()(const Point& p) const
    {
        const double x(p[0]), y(p[1]), z(p[2]);
        return a_*x*x + 2.0*b_*x*y + 2.0*c_*x*z + 2.0*d_*x
            +  e_*y*y + 2.0*f_*y*z + 2.0*g_*y
            +  h_*z*z + 2.0*i_*z
            +  j_;
    }

private:

    double a_, b_, c_, d_,
               e_, f_, g_,
                   h_, i_,
                       j_;
}; // clang-format on

struct PointCloudAdaptor
{
    using coord_t = Scalar;

    const std::vector<Point>& pcl; //!< A const ref to the data set origin

    /// The constructor that sets the data set source
    PointCloudAdaptor(const std::vector<Point>& pcl_) : pcl(pcl_) {}

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pcl.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline coord_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return pcl[idx][dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const
    {
        return false;
    }
};

using KDTree_t = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<Scalar, PointCloudAdaptor>, PointCloudAdaptor,
    3>;

std::vector<double> vertexAreas(const SurfaceMesh& mesh)
{
    std::vector<double> vertexAreas(mesh.n_vertices(), 0.);
    for (auto vit : mesh.vertices())
    {
        for (auto face : mesh.faces(vit))
        {
            const auto area = pmp::triangle_area(mesh, face);
            vertexAreas[vit.idx()] += area;
        }
    }

    return vertexAreas;
}

std::tuple<std::vector<Normal>, std::vector<Point>> smoothedNormalsAndCenters(
    float scale, const SurfaceMesh& mesh)
{
    std::vector<Normal> normals(mesh.n_faces());
    std::vector<Point> baryCenters(mesh.n_faces());
    for (auto face : mesh.faces())
    {
        normals[face.idx()] = SurfaceNormals::compute_face_normal(mesh, face);
        baryCenters[face.idx()] = pmp::centroid(mesh, face);
    }

    std::vector<Normal> fNormals(normals);
    PointCloudAdaptor pcl(baryCenters);

    KDTree_t tree(3, pcl,
                  nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();

    const auto gScale = 1.f / (2.f * scale * scale);

    for (auto face : mesh.faces())
    {
        std::vector<std::pair<unsigned int, Scalar>> neighborsScores;
        tree.radiusSearch(&baryCenters[face.idx()](0, 0), scale,
                          neighborsScores, nanoflann::SearchParams{});

        float w{0.f};
        for (const auto& ns : neighborsScores)
        {
            const auto score = exp(-ns.second * gScale);
            w += score;
            fNormals[face.idx()] += normals[ns.first] * score;
        }

        fNormals[face.idx()] /= w;
    }

    return {std::move(fNormals), std::move(baryCenters)};
}

std::vector<Quadric> computeInitialQuadrics(float normalsScale,
                                            const SurfaceMesh& mesh)
{
    std::vector<Quadric> quadrics(mesh.n_vertices());
    constexpr auto scale = 1. / 3.;

    std::cout << "-> Smooth normals" << std::endl;
    auto [normals, centers] = smoothedNormalsAndCenters(normalsScale, mesh);

    std::cout << "-> Compute quadrics" << std::endl;

    // compute quadrics per vertex
    for (auto vit : mesh.vertices())
    {
        for (auto face : mesh.faces(vit))
        {
            quadrics[vit.idx()] +=
                Quadric(normals[face.idx()], centers[face.idx()]);
            // quadrics[vit.idx()] *= pmp::triangle_area(mesh, face);
        }
        quadrics[vit.idx()] *= scale;
    }

    return quadrics;
}

void computeQuadricField(Scalar scale, const KDTree_t& tree,
                         const SurfaceMesh& mesh,
                         const std::vector<double>& areas,
                         const std::vector<Quadric>& inputQuadrics,
                         std::vector<Quadric>& outputQuadrics)
{
    (void)areas;
    outputQuadrics = inputQuadrics;
    const auto radius = 2. * scale;
    const auto gScale = 1. / (2. * scale * scale);

    for (auto vit : mesh.vertices())
    {
        const auto& pos = mesh.position(vit);
        std::vector<std::pair<unsigned int, Scalar>> neighborsScores;
        tree.radiusSearch(&pos(0, 0), radius, neighborsScores,
                          nanoflann::SearchParams{});

        double w{0.};

        for (const auto& ns : neighborsScores)
        {
            Scalar score = static_cast<double>(
                exp(-ns.second * gScale)); // * areas[ns.first]);
            w += score;
            outputQuadrics[vit.idx()] += inputQuadrics[ns.first] * score;
        }

        outputQuadrics[vit.idx()] *= 1. / w;
    }
}

void diffuseQuadricField(Scalar spatialScale, Scalar rangeScale,
                         const KDTree_t& tree, const SurfaceMesh& mesh,
                         const std::vector<double>& areas,
                         const std::vector<Quadric>& inputQuadrics,
                         std::vector<Quadric>& outputQuadrics)
{
    (void)areas;
    outputQuadrics = inputQuadrics;
    const auto radius = 2. * spatialScale;
    const auto gSpatialScale = 1. / (2. * spatialScale * spatialScale);
    const auto gRangeScale = 1. / (2. * rangeScale * rangeScale);

    for (auto vit : mesh.vertices())
    {
        const auto& pos = mesh.position(vit);
        std::vector<std::pair<unsigned int, Scalar>> neighborsScores;
        tree.radiusSearch(&pos(0, 0), radius, neighborsScores,
                          nanoflann::SearchParams{});

        double w{0.};

        for (const auto& ns : neighborsScores)
        {
            const auto qError2 = inputQuadrics[ns.first](pos);

            Scalar score = static_cast<double>(
                exp(-(ns.second * gSpatialScale +
                      gRangeScale * qError2))); // *areas[ns.first]);
            w += score;
            outputQuadrics[vit.idx()] += inputQuadrics[ns.first] * score;
        }

        outputQuadrics[vit.idx()] *= (1. / w);
    }
}

void updateVertices(const std::vector<Quadric>& filteredQuadricField,
                    SurfaceMesh& mesh)
{
    for (auto vit : mesh.vertices())
    {
        auto& pos = mesh.position(vit);
        pos = filteredQuadricField[vit.idx()].solve();
    }
}

} // namespace

int main(int argc, char** argv)
{
    SurfaceMesh mesh;

    std::cout << "Loading mesh" << std::endl;

    if (argc < 3 || argc > 6)
    {
        std::cout << "Invalid inputs" << std::endl;
        return -1;
    }

    std::cout << "Input mesh: " << argv[1] << std::endl;
    std::cout << "Output mesh: " << argv[2] << std::endl;

    Scalar sigmab = 0.0005f;
    Scalar sigmas = 0.001f;
    Scalar sigmar = 0.001f;

    if (argc == 4)
    {
        sigmas = static_cast<Scalar>(std::atof(argv[3]));
        sigmab = sigmas / 2;
    }
    else if (argc == 5)
    {
        sigmas = static_cast<Scalar>(std::atof(argv[3]));
        sigmar = static_cast<Scalar>(std::atof(argv[4]));
        sigmab = sigmas / 2;
    }
    else if (argc == 6)
    {
        sigmas = static_cast<Scalar>(std::atof(argv[3]));
        sigmar = static_cast<Scalar>(std::atof(argv[4]));
        sigmab = static_cast<Scalar>(std::atof(argv[5]));
    }

    std::cout << "Sigma_b: " << sigmab << std::endl;
    std::cout << "Sigma_s: " << sigmas << std::endl;
    std::cout << "Sigma_r: " << sigmar << std::endl;

    mesh.read(argv[1]);

    std::cout << "Compute vertices area" << std::endl;
    const auto areas = vertexAreas(mesh);

    std::cout << "Compute initial quadrics" << std::endl;
    auto quadrics = computeInitialQuadrics(sigmas / 4.f, mesh);

    std::cout << "Build kd-tree" << std::endl;
    PointCloudAdaptor pcl(mesh.positions());
    KDTree_t tree(3, pcl, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();

    std::cout << "Compute quadric field" << std::endl;

    std::vector<Quadric> quadricField;
    computeQuadricField(sigmab, tree, mesh, areas, quadrics, quadricField);

    std::cout << "Diffuse quadric field" << std::endl;
    std::vector<Quadric> filteredQuadricField;
    diffuseQuadricField(sigmas, sigmar, tree, mesh, areas, quadricField,
                        filteredQuadricField);

    std::cout << "Update vertices using filtered quadrics" << std::endl;
    updateVertices(filteredQuadricField, mesh);
    //updateVertices(quadrics, mesh);

    std::cout << "Write mesh" << std::endl;
    mesh.write(argv[2]);

    return 0;
}
