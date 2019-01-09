 function plotConstellation(srx)

            colors;
            scatplot(real(srx),imag(srx),'voronoi');
%             plot(srx, '.', 'color', blue, 'MarkerSize',3);
            hold on
            if nargin >1
                % Plot clusters centroids
                c = varargin{2};
                plot(c,  'o', 'color', red, 'MarkerFaceColor', red, 'MarkerSize',3)
            end
            xlim(1.1*[min(real(srx)) max(real(srx))])
            ylim(1.1*[min(imag(srx)) max(imag(srx))])
end