/**
 * Batch computation of JB2008 atmospheric density along a satellite track.
 *
 * Two methods are provided with different approaches to supplying the solar
 * and geomagnetic indices required by the JB2008 model:
 *
 * <ul>
 *   <li>{@link #getDensity} — caller supplies all nine space weather index
 *       arrays (F10, F10B, S10, S10B, XM10, XM10B, Y10, Y10B, DSTDTC).
 *       Values are linearly interpolated between the supplied grid points, so
 *       daily indices pre-interpolated to satellite epochs work directly.
 *       Use this when you need dates outside the bundled Orekit data range,
 *       or when you want to use a custom / updated space weather dataset.</li>
 *
 *   <li>{@link #getDensityFromOrekitData} — no space weather arrays needed.
 *       Reads the SOLFSMY and DTCFILE files from the Orekit data context
 *       configured by {@code setup_orekit_data} at JVM startup. Simpler to
 *       call, but limited to the date range covered by the bundled data
 *       (typically a few years behind the current date).</li>
 * </ul>
 *
 * Both methods accept positions either as ECEF Cartesian [x, y, z] in metres
 * or as geodetic [lat_deg, lon_deg, alt_m] (WGS-84). Pass {@code geodetic=true}
 * to use the geodetic form; latitude and longitude must be in degrees.
 *
 * @author Kyle Murphy, kylemurphy.spacephys@gmail.com
 * @version 1.2
 * @since 2026-05-28
 */

package org.contigo.orekit_utils;

import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;
import org.orekit.bodies.CelestialBody;
import org.orekit.bodies.CelestialBodyFactory;
import org.orekit.bodies.GeodeticPoint;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.models.earth.atmosphere.JB2008;
import org.orekit.models.earth.atmosphere.JB2008InputParameters;
import org.orekit.models.earth.atmosphere.data.JB2008SpaceEnvironmentData;
import org.hipparchus.geometry.euclidean.threed.Vector3D;

public class JB2008DensityBatchHelper {

    private static final Frame ECEF =
            FramesFactory.getITRF(IERSConventions.IERS_2010, true);

    private static final CelestialBody SUN =
            CelestialBodyFactory.getSun();

    private static final OneAxisEllipsoid EARTH =
            new OneAxisEllipsoid(
                    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                    Constants.WGS84_EARTH_FLATTENING,
                    ECEF);

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Array-backed JB2008InputParameters with linear interpolation.
     *
     * The offsets array defines the interpolation grid (seconds since
     * referenceDate). All parameter arrays must have the same length.
     */
    private static class LinearJB2008Params implements JB2008InputParameters {

        private final AbsoluteDate refDate;
        private final double[] offsets;
        private final double[] f10, f10b;
        private final double[] s10, s10b;
        private final double[] xm10, xm10b;
        private final double[] y10, y10b;
        private final double[] dstdtc;

        LinearJB2008Params(AbsoluteDate refDate, double[] offsets,
                           double[] f10,   double[] f10b,
                           double[] s10,   double[] s10b,
                           double[] xm10,  double[] xm10b,
                           double[] y10,   double[] y10b,
                           double[] dstdtc) {
            this.refDate = refDate;
            this.offsets = offsets;
            this.f10     = f10;   this.f10b   = f10b;
            this.s10     = s10;   this.s10b   = s10b;
            this.xm10    = xm10;  this.xm10b  = xm10b;
            this.y10     = y10;   this.y10b   = y10b;
            this.dstdtc  = dstdtc;
        }

        private double interp(AbsoluteDate date, double[] vals) {
            double t = date.durationFrom(refDate);
            int n = offsets.length;
            if (n == 1 || t <= offsets[0])    return vals[0];
            if (t >= offsets[n - 1])           return vals[n - 1];
            // unsigned-shift right to avoid overflow on (lo + hi)
            int lo = 0, hi = n - 2;
            while (lo < hi) {
                int mid = (lo + hi) >>> 1;
                if (offsets[mid + 1] <= t) lo = mid + 1;
                else                        hi = mid;
            }
            double frac = (t - offsets[lo]) / (offsets[lo + 1] - offsets[lo]);
            return vals[lo] + frac * (vals[lo + 1] - vals[lo]);
        }

        @Override public AbsoluteDate getMinDate() { return refDate.shiftedBy(offsets[0]); }
        @Override public AbsoluteDate getMaxDate() { return refDate.shiftedBy(offsets[offsets.length - 1]); }

        @Override public double getF10(AbsoluteDate d)    { return interp(d, f10);    }
        @Override public double getF10B(AbsoluteDate d)   { return interp(d, f10b);   }
        @Override public double getS10(AbsoluteDate d)    { return interp(d, s10);    }
        @Override public double getS10B(AbsoluteDate d)   { return interp(d, s10b);   }
        @Override public double getXM10(AbsoluteDate d)   { return interp(d, xm10);   }
        @Override public double getXM10B(AbsoluteDate d)  { return interp(d, xm10b);  }
        @Override public double getY10(AbsoluteDate d)    { return interp(d, y10);    }
        @Override public double getY10B(AbsoluteDate d)   { return interp(d, y10b);   }
        @Override public double getDSTDTC(AbsoluteDate d) { return interp(d, dstdtc); }
    }

    /**
     * Convert one row of the positions array to an ECEF Vector3D.
     *
     * @param pos      one row from the caller's positions array
     * @param geodetic if true, pos = [lat_deg, lon_deg, alt_m] (WGS-84);
     *                 if false, pos = [x_m, y_m, z_m] (ECEF Cartesian)
     */
    private static Vector3D positionVector(double[] pos, boolean geodetic) {
        if (geodetic) {
            return EARTH.transform(new GeodeticPoint(
                    Math.toRadians(pos[0]),   // latitude  (degrees → radians)
                    Math.toRadians(pos[1]),   // longitude (degrees → radians)
                    pos[2]));                 // altitude  (metres, unchanged)
        }
        return new Vector3D(pos[0], pos[1], pos[2]);
    }

    /**
     * Shared density loop used by all public methods.
     */
    private static double[] computeDensity(
            JB2008 atmosphere,
            AbsoluteDate referenceDate,
            double[] offsetSeconds,
            double[][] positions,
            boolean geodetic) {

        double[] density = new double[offsetSeconds.length];
        for (int i = 0; i < offsetSeconds.length; i++) {
            AbsoluteDate date = referenceDate.shiftedBy(offsetSeconds[i]);
            Vector3D pos = positionVector(positions[i], geodetic);
            density[i] = atmosphere.getDensity(date, pos, ECEF);
        }
        return density;
    }

    // -------------------------------------------------------------------------
    // Public API — user-supplied space weather indices
    // -------------------------------------------------------------------------

    /**
     * Compute JB2008 density with caller-supplied space weather indices.
     * Positions are interpreted as ECEF Cartesian [x, y, z] in metres.
     *
     * @param referenceDate  epoch corresponding to offsetSeconds[0] == 0
     * @param offsetSeconds  (N) seconds since referenceDate for each state
     * @param positions      (N,3) ECEF positions [x, y, z] in metres
     * @param f10            (N) F10 solar flux index (1-day lag)
     * @param f10b           (N) 81-day centred average of F10
     * @param s10            (N) S10 solar EUV proxy index
     * @param s10b           (N) 81-day centred average of S10
     * @param xm10           (N) XM10 solar MUV proxy index
     * @param xm10b          (N) 81-day centred average of XM10
     * @param y10            (N) Y10 solar X-ray proxy index
     * @param y10b           (N) 81-day centred average of Y10
     * @param dstdtc         (N) exospheric temperature correction (K)
     * @return double[N] density in kg/m³ at each epoch
     */
    public static double[] getDensity(
            AbsoluteDate referenceDate,
            double[] offsetSeconds,
            double[][] positions,
            double[] f10, double[] f10b,
            double[] s10, double[] s10b,
            double[] xm10, double[] xm10b,
            double[] y10, double[] y10b,
            double[] dstdtc) {

        return getDensity(referenceDate, offsetSeconds, positions, false,
                f10, f10b, s10, s10b, xm10, xm10b, y10, y10b, dstdtc);
    }

    /**
     * Compute JB2008 density with caller-supplied space weather indices.
     *
     * @param referenceDate  epoch corresponding to offsetSeconds[0] == 0
     * @param offsetSeconds  (N) seconds since referenceDate for each state
     * @param positions      (N,3) satellite state array; interpretation depends
     *                       on {@code geodetic}
     * @param geodetic       if {@code true}, positions columns are
     *                       [lat_deg, lon_deg, alt_m] (WGS-84, degrees);
     *                       if {@code false}, columns are [x_m, y_m, z_m] ECEF
     * @param f10            (N) F10 solar flux index (1-day lag)
     * @param f10b           (N) 81-day centred average of F10
     * @param s10            (N) S10 solar EUV proxy index
     * @param s10b           (N) 81-day centred average of S10
     * @param xm10           (N) XM10 solar MUV proxy index
     * @param xm10b          (N) 81-day centred average of XM10
     * @param y10            (N) Y10 solar X-ray proxy index
     * @param y10b           (N) 81-day centred average of Y10
     * @param dstdtc         (N) exospheric temperature correction (K)
     * @return double[N] density in kg/m³ at each epoch
     */
    public static double[] getDensity(
            AbsoluteDate referenceDate,
            double[] offsetSeconds,
            double[][] positions,
            boolean geodetic,
            double[] f10, double[] f10b,
            double[] s10, double[] s10b,
            double[] xm10, double[] xm10b,
            double[] y10, double[] y10b,
            double[] dstdtc) {

        if (positions.length != offsetSeconds.length) {
            throw new IllegalArgumentException(
                    "offsetSeconds and positions must have the same length");
        }

        LinearJB2008Params params = new LinearJB2008Params(
                referenceDate, offsetSeconds,
                f10, f10b, s10, s10b, xm10, xm10b, y10, y10b, dstdtc);

        return computeDensity(new JB2008(params, SUN, EARTH),
                referenceDate, offsetSeconds, positions, geodetic);
    }

    // -------------------------------------------------------------------------
    // Public API — Orekit-managed space weather
    // -------------------------------------------------------------------------

    /**
     * Compute JB2008 density using space weather data bundled with the Orekit
     * data archive. Positions are interpreted as ECEF Cartesian [x, y, z] in metres.
     *
     * <p><b>Limitation:</b> the bundled data files have a fixed date range
     * (typically ending a year or two before the Orekit data zip was published).
     * Requesting epochs outside that range will throw an {@code OrekitException}.
     * Use {@link #getDensity} with manually supplied indices for more recent or
     * custom date ranges.
     *
     * @param referenceDate  epoch corresponding to offsetSeconds[0] == 0
     * @param offsetSeconds  (N) seconds since referenceDate for each state
     * @param positions      (N,3) ECEF positions [x, y, z] in metres
     * @return double[N] density in kg/m³ at each epoch
     */
    public static double[] getDensityFromOrekitData(
            AbsoluteDate referenceDate,
            double[] offsetSeconds,
            double[][] positions) {

        return getDensityFromOrekitData(referenceDate, offsetSeconds, positions, false);
    }

    /**
     * Compute JB2008 density using space weather data bundled with the Orekit
     * data archive.
     *
     * <p><b>Limitation:</b> the bundled data files have a fixed date range
     * (typically ending a year or two before the Orekit data zip was published).
     * Requesting epochs outside that range will throw an {@code OrekitException}.
     * Use {@link #getDensity} with manually supplied indices for more recent or
     * custom date ranges.
     *
     * @param referenceDate  epoch corresponding to offsetSeconds[0] == 0
     * @param offsetSeconds  (N) seconds since referenceDate for each state
     * @param positions      (N,3) satellite state array; interpretation depends
     *                       on {@code geodetic}
     * @param geodetic       if {@code true}, positions columns are
     *                       [lat_deg, lon_deg, alt_m] (WGS-84, degrees);
     *                       if {@code false}, columns are [x_m, y_m, z_m] ECEF
     * @return double[N] density in kg/m³ at each epoch
     */
    public static double[] getDensityFromOrekitData(
            AbsoluteDate referenceDate,
            double[] offsetSeconds,
            double[][] positions,
            boolean geodetic) {

        if (positions.length != offsetSeconds.length) {
            throw new IllegalArgumentException(
                    "offsetSeconds and positions must have the same length");
        }

        JB2008SpaceEnvironmentData params = new JB2008SpaceEnvironmentData(
                JB2008SpaceEnvironmentData.DEFAULT_SUPPORTED_NAMES_SOLFSMY,
                JB2008SpaceEnvironmentData.DEFAULT_SUPPORTED_NAMES_DTC);

        return computeDensity(new JB2008(params, SUN, EARTH),
                referenceDate, offsetSeconds, positions, geodetic);
    }
}
